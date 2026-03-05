from typing import *
import torch
from torch.autograd import Function
from tqdm import tqdm
from . import Algorithm
from .. import spconv, utils
from ... import kernels


def _partition_octree(spatial_coords, max_voxels, depth=0):
    """
    Recursively split spatial extent along longest axis until each
    partition has <= max_voxels. Returns list of (tile_min, tile_max) tensors.
    tile_min is inclusive, tile_max is exclusive.
    """
    N = spatial_coords.shape[0]
    if N <= max_voxels:
        tile_min = spatial_coords.min(0)[0]
        tile_max = spatial_coords.max(0)[0] + 1
        return [(tile_min, tile_max)]

    mins = spatial_coords.min(0)[0]
    maxs = spatial_coords.max(0)[0]
    extents = maxs - mins
    axis = extents.argmax().item()

    if extents[axis] < 2:
        return [(mins, maxs + 1)]

    mid = (mins[axis].item() + maxs[axis].item()) // 2 + 1
    left_mask = spatial_coords[:, axis] < mid
    right_mask = ~left_mask
    left_count = left_mask.sum().item()
    right_count = right_mask.sum().item()

    tiles = []
    if left_count > 0:
        tiles.extend(_partition_octree(spatial_coords[left_mask], max_voxels, depth + 1))
    if right_count > 0:
        tiles.extend(_partition_octree(spatial_coords[right_mask], max_voxels, depth + 1))
    return tiles


class SubMConv3dNeighborCache:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def compute_kernel_idx(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg
        
    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_{block_size}']
    
    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if not hasattr(self, f'valid_kernel_seg_{block_size}'):
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_seg_{block_size}']


class SubMConv3dFunction(Function):
    @staticmethod
    def tiled_forward(feats, coords, shape, weight, bias, kernel_size, dilation,
                      max_voxels_per_tile=1_000_000):
        """
        Spatially-tiled submanifold conv. Splits voxels into tiles with halo
        overlap, runs conv per tile, merges interior results.
        Mathematically identical to full conv — submanifold conv is local.

        VRAM strategy: input feats and output are kept on CPU during tiling.
        Only the per-tile subset (feats, cache, conv output) lives on GPU
        transiently. The final output is copied to GPU at the end.

        Accepts feats on CPU or GPU. Uses weight.device as compute device.
        """
        N = feats.shape[0]
        Co = weight.shape[0]
        Ci = weight.shape[-1]
        device = weight.device  # always GPU — feats may be on CPU

        _ma = torch.cuda.memory_allocated
        feats_mb = feats.numel() * feats.element_size() / 1024**2
        out_mb = N * Co * feats.element_size() / 1024**2
        print(f"[tiled_conv] N={N:,} Ci={Ci} Co={Co} K={kernel_size} "
              f"feats={feats_mb:.0f}MB out={out_mb:.0f}MB "
              f"feats_dev={feats.device} alloc={_ma()//1048576}MB", flush=True)

        if N <= max_voxels_per_tile:
            feats_gpu = feats.to(device) if not feats.is_cuda else feats
            cache = SubMConv3dFunction._compute_neighbor_cache(
                coords, shape, kernel_size, dilation)
            out = SubMConv3dFunction._sparse_submanifold_conv_forward(
                feats_gpu, cache, weight, bias)
            del cache, feats_gpu
            return out

        # Halo radius = max kernel radius * dilation per axis
        halo = max((k // 2) * d for k, d in zip(kernel_size, dilation))
        spatial = coords[:, 1:]  # [N, 3] — drop batch dim

        # Partition into tiles (suppress per-leaf prints)
        tiles = _partition_octree(spatial, max_voxels_per_tile)

        # Ensure feats are on CPU for tiling
        if feats.is_cuda:
            feats_cpu = feats.cpu()
            del feats
            torch.cuda.empty_cache()
        else:
            feats_cpu = feats

        vram_before = _ma() // 1048576
        print(f"[tiled_conv] {len(tiles)} tiles, VRAM baseline={vram_before}MB", flush=True)

        # Accumulate output on CPU — avoids large GPU allocation during tiling
        out_cpu = torch.empty((N, Co), device='cpu', dtype=feats_cpu.dtype)
        total_interior = 0
        peak_tile_vram = 0

        pbar = tqdm(tiles, desc="Tiled Conv3d", unit="tile",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for i, (tile_min, tile_max) in enumerate(pbar):
            # Interior: voxels strictly within tile bounds
            interior_mask = ((spatial >= tile_min) & (spatial < tile_max)).all(dim=1)

            # Halo: voxels within halo distance of tile boundary
            halo_min = tile_min - halo
            halo_max = tile_max + halo
            halo_mask = ((spatial >= halo_min) & (spatial < halo_max)).all(dim=1)

            # Extract tile tensors — feats from CPU, coords already on GPU
            tile_indices = halo_mask.nonzero(as_tuple=True)[0]
            tile_indices_cpu = tile_indices.cpu()
            tile_feats = feats_cpu[tile_indices_cpu].to(device)
            tile_coords = coords[tile_indices].contiguous()
            interior_count = interior_mask.sum().item()
            total_interior += interior_count

            # Compute neighbor cache for this tile only
            tile_cache = SubMConv3dFunction._compute_neighbor_cache(
                tile_coords, shape, kernel_size, dilation)

            # Run conv on tile
            tile_out = SubMConv3dFunction._sparse_submanifold_conv_forward(
                tile_feats, tile_cache, weight, bias)

            torch.cuda.synchronize()
            tile_vram = _ma() // 1048576
            peak_tile_vram = max(peak_tile_vram, tile_vram)

            # Map interior voxels back — scatter to CPU output
            interior_in_tile = interior_mask[tile_indices]
            interior_global_cpu = tile_indices_cpu[interior_in_tile.cpu()]
            out_cpu[interior_global_cpu] = tile_out[interior_in_tile].cpu()

            del tile_feats, tile_coords, tile_cache, tile_out, tile_indices, tile_indices_cpu
            del interior_mask, halo_mask, interior_in_tile, interior_global_cpu
            torch.cuda.empty_cache()

            pbar.set_postfix_str(f"vram={_ma()//1048576}MB peak={peak_tile_vram}MB")

        pbar.close()
        del feats_cpu

        # Move final output to GPU
        out = out_cpu.to(device)
        del out_cpu

        vram_final = _ma() // 1048576
        print(f"[tiled_conv] done: {total_interior}/{N} voxels, "
              f"tile_peak={peak_tile_vram}MB, final={vram_final}MB", flush=True)

        if total_interior != N:
            print(f"[tiled_conv] WARNING: missing {N - total_interior} voxels!", flush=True)

        return out

    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> SubMConv3dNeighborCache:
        assert coords.is_contiguous(), "Coords should be contiguous"
        assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
        N, C, W, H, D = shape
        
        hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device)

        if spconv.ALGORITHM in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
                    hashmap_keys, hashmap_vals, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            return SubMConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
            })
        
        elif spconv.ALGORITHM in [Algorithm.MASKED_IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
                    hashmap_keys, hashmap_vals, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            V = kernel_size[0] * kernel_size[1] * kernel_size[2]
            assert V <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
            
            gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
                kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
            
            return SubMConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
                'gray_code': gray_code,
                'sorted_idx': sorted_idx,
                'valid_signal_seg': valid_signal_seg,
                'valid_signal_i': valid_signal_i,
                'valid_signal_o': valid_signal_o,
            })
                
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

    def _compute_neighbor_cache_torch(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int]
    ) -> SubMConv3dNeighborCache:
        assert spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM, "Only explicit_gemm is supported for torch implementation"
        N, C, W, H, D = shape
        L = coords.shape[0]
        assert N * W * H * D <= 2**32, "Currently, the max number of elements in a tensor is 2^32"
        M = torch.tensor([W * H * D, H * D, D, 1], device=coords.device).int()
        
        keys = (coords * M[None]).sum(dim=-1)
        sorted_keys, indices = torch.sort(keys)
        
        # Compute neighbor coords
        offset = torch.meshgrid(
            torch.arange(-(kernel_size[0] // 2) * dilation[0], kernel_size[0] // 2 * dilation[0] + 1, dilation[0]),
            torch.arange(-(kernel_size[1] // 2) * dilation[1], kernel_size[1] // 2 * dilation[1] + 1, dilation[1]),
            torch.arange(-(kernel_size[2] // 2) * dilation[2], kernel_size[2] // 2 * dilation[2] + 1, dilation[2]),
            indexing='ij'
        )
        offset = torch.stack(offset, dim=-1).reshape(-1, 3).int().to(coords.device)
        neighbor_coords = coords.unsqueeze(1).repeat(1, kernel_size[0] * kernel_size[1] * kernel_size[2], 1)
        neighbor_coords[:, :, -3:] += offset.unsqueeze(0)                                    # [N, kernel_vol, 4]
        neighbor_coords = neighbor_coords.reshape(-1, 4)                                    # [N * kernel_vol, 4]
        neighbor_valid = (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < W) & \
                         (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < H) & \
                         (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < D)
        neighbor_keys = (neighbor_coords * M[None]).sum(dim=-1)
        neighbor_search_indices = torch.searchsorted(sorted_keys, neighbor_keys)
        neighbor_search_indices = torch.clamp(neighbor_search_indices, 0, sorted_keys.shape[0] - 1)
        neighbor_valid &= sorted_keys[neighbor_search_indices] == neighbor_keys
        neighbor_indices = torch.full((L * kernel_size[0] * kernel_size[1] * kernel_size[2],), 0xffffffff, dtype=torch.long, device=coords.device)
        neighbor_indices[neighbor_valid] = indices[neighbor_search_indices[neighbor_valid]]
        return SubMConv3dNeighborCache(**{'neighbor_map': neighbor_indices.reshape(L, -1).to(torch.uint32)})
        
    @staticmethod
    def _sparse_submanifold_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous"
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw
        
        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:        
            neighbor_map = neighbor_cache['neighbor_map']
            
            # im2col
            im2col = torch.zeros((N * V, Ci), device=feats.device, dtype=feats.dtype)
            mask = neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)
            
            # addmm
            weight = weight.view(Co, V * Ci).transpose(0, 1)
            if bias is not None:
                output = torch.addmm(bias, im2col, weight)
            else:
                output = torch.mm(im2col, weight)
        
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            output = kernels.triton.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback
            )
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return output

    @staticmethod
    def _sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache['neighbor_map']
            
            if feats.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
                inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)
                
                # addmm
                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
                
            if weight.requires_grad:
                # im2col
                im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)
                
                # addmm
                grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
        
        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache['neighbor_map'],
                neighbor_cache['sorted_idx'],
                neighbor_cache['valid_kernel_callback'],
                neighbor_cache['valid_kernel_seg_callback'],
                neighbor_cache['valid_signal_i'],
                neighbor_cache['valid_signal_o'],
                neighbor_cache['valid_signal_seg']
            )
            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)
            
        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        neighbor_cache: Optional[SubMConv3dNeighborCache],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        
        # check if neighbor map is already computed
        if neighbor_cache is None:
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), dilation)
            
        # compute output
        output = SubMConv3dFunction._sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias)
        
        # save for backward
        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        
        return output, neighbor_cache
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        
        grad_input, grad_weight, grad_bias = SubMConv3dFunction._sparse_submanifold_conv_backward(grad_output, feats, neighbor_cache, weight, bias)
        
        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None


def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SubMConv3dNeighborCache] = None,
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
    """
    Sparse submanifold convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SubMConv3dNeighborCache]): neighbor cache for forward.
            if None, will be computed in forward.
        dilation (Tuple[int, int, int]): dilation rate.

    Returns:
        Tuple[torch.Tensor, SubMConv3dNeighborCache]:
            - output (torch.Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward.
    """
    return SubMConv3dFunction.apply(feats, coords, shape, neighbor_cache, weight, bias, dilation)
