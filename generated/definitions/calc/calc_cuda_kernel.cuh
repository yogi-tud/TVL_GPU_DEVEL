#pragma once

#include "../../../utils.cuh"
#include "../../../cuda_try.cuh"

#define CUDA_WARP_SIZE 32

// kernels
template <typename T, T (*OP)(T, T)>
__global__ void
elementwise_op_kernel(T* vec_a, T* vec_b, T* vec_c, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t pos = tid; pos < element_count; pos += blockDim.x * gridDim.x) {
        vec_c[pos] = OP(vec_a[pos], vec_b[pos]);
    }
}

template <typename REGISTER_TYPE, typename REGISTER_TYPE::element_type::base_type (*OP)(typename REGISTER_TYPE::element_type::base_type, typename REGISTER_TYPE::element_type::base_type)>
REGISTER_TYPE launch_elemenwise_op(REGISTER_TYPE vec_a, REGISTER_TYPE vec_b, size_t vector_size_bits)
{
    REGISTER_TYPE vec_c = std::make_shared<typename REGISTER_TYPE::element_type>();
    size_t element_count = vector_size_bits / (sizeof(typename REGISTER_TYPE::element_type::base_type) * 8);
    launch_dimensions dim = get_launch_dimensions(element_count);
    elementwise_op_kernel<typename REGISTER_TYPE::element_type::base_type, OP>
        <<<dim.grid_size, dim.block_size>>>(vec_a->data, vec_b->data, vec_c->data, element_count);
    return vec_c;
}

// shift left by using negative dist
template <typename T>
__global__ void
shift_right_kernel(T* vec_a, T* vec_tmp, int dist, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t pos = tid; pos < element_count; pos += blockDim.x * gridDim.x) {
        vec_tmp[pos] = vec_a[(element_count + pos - dist) % element_count];
    }
}

template <typename T>
__global__ void invert_kernel(T* vec_a, T* vec_res, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t pos = tid; pos < element_count; pos += blockDim.x * gridDim.x) {
        vec_res[pos] = ~vec_a[pos];
    }
}

template <typename T>
__global__ void hadd_kernel(T* vec_a, T* vec_c, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T sum;
    for (size_t pos = tid; pos < element_count; pos += blockDim.x * gridDim.x) {
        atomicAdd(sum, vec_a[tid]);
    }
    atomicAdd(vec_c, sum);
}

template <int workaround = 1> // silence ODR
__global__ void kernel_3pass_popc_none_striding(
    uint8_t* mask, uint32_t* pss, uint32_t chunk_length32,
    uint32_t element_count)
{
    uint32_t tid =
        (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    uint32_t chunk_count = ceildiv(element_count, chunk_length32 * 32);
    for (; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = chunk_length32 * 4 *
                       tid; // index for 1st 8bit-element of this chunk
        uint32_t bit_idx = idx * 8;
        // assuming chunk_length to be multiple of 32
        uint32_t remaining_bytes_for_grid = (element_count - bit_idx) / 8;
        uint32_t bytes_to_process = chunk_length32 * 4;
        if (remaining_bytes_for_grid < bytes_to_process) {
            bytes_to_process = remaining_bytes_for_grid;
        }
        // assuming chunk_length to be multiple of 32
        uint32_t popcount = 0;
        int i = 0;
        for (; i < bytes_to_process / 4 * 4; i += 4) {
            popcount += __popc(*reinterpret_cast<uint32_t*>(mask + idx + i));
        }
        if (i < bytes_to_process / 2 * 2) {
            popcount += __popc(*reinterpret_cast<uint16_t*>(mask + idx + i));
            i += 2;
        }
        if (i < bytes_to_process)
            popcount += __popc(*reinterpret_cast<uint8_t*>(mask + idx + i));
        pss[tid] = popcount;
    }
}

template <int workaround = 1> // silence ODR
__global__ void kernel_3pass_pss_gmem_monolithic(
    uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t stride = (1 << depth);
    tid = 2 * tid * stride + stride - 1;
    // tid is element id

    // thread loads element at tid and tid+stride
    if (tid >= chunk_count) {
        return;
    }
    uint32_t left_e = pss[tid];
    if (tid + stride < chunk_count) {
        pss[tid + stride] += left_e;
    }
    else {
        (*out_count) += left_e;
    }
}

__device__ uint32_t d_3pass_pproc_pssidx(
    uint32_t thread_idx, uint32_t* pss, uint32_t chunk_count_p2)
{
    chunk_count_p2 /= 2; // start by trying the subtree with length of half the
                         // next rounded up power of 2 of chunk_count
    uint32_t consumed = 0; // length of subtrees already fit inside idx_acc
    uint32_t idx_acc = 0; // assumed starting position for this chunk
    while (chunk_count_p2 >= 1) {
        if (thread_idx >= consumed + chunk_count_p2) {
            // partial tree [consumed, consumed+chunk_count_p2] fits into left
            // side of thread_idx
            idx_acc += pss[consumed + chunk_count_p2 - 1];
            consumed += chunk_count_p2;
        }
        chunk_count_p2 /= 2;
    }
    return idx_acc;
}

template <int workaround = 1> // silence ODR
__global__ void kernel_3pass_pss2_gmem_monolithic(
    uint32_t* pss_in, uint32_t* pss_out, uint32_t chunk_count,
    uint32_t chunk_count_p2)
{
    uint32_t tid =
        (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    if (tid >= chunk_count) {
        return;
    }
    pss_out[tid] = d_3pass_pproc_pssidx(tid, pss_in, chunk_count_p2);
}

template <uint32_t BLOCK_DIM, typename T>
__global__ void kernel_3pass_proc_true_striding_optimized_writeout(
    T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t* popc,
    size_t chunk_length, size_t element_count, uint32_t chunk_count_p2,
    uint32_t* offset)
{
    uint32_t mask_byte_count = element_count / 8;
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    if (offset != NULL) {
        output += *offset;
    }
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_DIM / CUDA_WARP_SIZE;
    __shared__ uint32_t smem[BLOCK_DIM];
    __shared__ uint32_t out_indices[BLOCK_DIM];
    __shared__ uint32_t smem_out_idx[WARPS_PER_BLOCK];
    uint32_t warp_remainder = WARPS_PER_BLOCK;
    while (warp_remainder % 2 == 0) {
        warp_remainder /= 2;
    }
    if (warp_remainder == 0) {
        warp_remainder = 1;
    }
    uint32_t grid_stride = chunk_length * warp_remainder;
    while (grid_stride % (CUDA_WARP_SIZE * BLOCK_DIM) != 0 ||
           grid_stride * gridDim.x < element_count ||
           grid_stride / WARPS_PER_BLOCK < chunk_length) {
        grid_stride *= 2;
    }
    uint32_t warp_stride = grid_stride / WARPS_PER_BLOCK;
    uint32_t warp_offset = threadIdx.x % CUDA_WARP_SIZE;
    uint32_t warp_index = threadIdx.x / CUDA_WARP_SIZE;
    uint32_t warp_base_index = threadIdx.x - warp_offset;
    uint32_t base_idx = blockIdx.x * grid_stride + warp_index * warp_stride;
    uint32_t stride = 1024;
    if (base_idx >= element_count) return;
    if (warp_offset == 0) {
        smem_out_idx[warp_index] = pss[base_idx / chunk_length];
    }
    __syncwarp();
    uint32_t warp_output_index = smem_out_idx[warp_index];
    uint32_t stop_idx = base_idx + warp_stride;
    if (stop_idx > element_count) {
        stop_idx = element_count;
    }
    uint32_t elements_aquired = 0;
    while (base_idx < stop_idx) {
        // check chunk popcount at base_idx for potential skipped
        if (popc) {
            if (chunk_length >= stride) {
                if (popc[base_idx / chunk_length] == 0) {
                    base_idx += chunk_length;
                    continue;
                }
            }
            else {
                bool empty_stride = true;
                for (uint32_t cid = base_idx / chunk_length;
                     cid < (base_idx + stride) / chunk_length; cid++) {
                    if (popc[cid] != 0) {
                        empty_stride = false;
                        break;
                    }
                }
                if (empty_stride) {
                    base_idx += stride;
                    continue;
                }
            }
        }
        uint32_t mask_idx = base_idx / 8 + warp_offset * 4;
        if (mask_idx < mask_byte_count) {
            uchar4 ucx = {0, 0, 0, 0};
            if (mask_idx + 4 > mask_byte_count) {
                switch (mask_byte_count - mask_idx) {
                    case 3: ucx.z = *(mask + mask_idx + 2);
                    case 2: ucx.y = *(mask + mask_idx + 1);
                    case 1: ucx.x = *(mask + mask_idx);
                }
            }
            else {
                ucx = *reinterpret_cast<uchar4*>(mask + mask_idx);
            }
            uchar4 uix{ucx.w, ucx.z, ucx.y, ucx.x};
            smem[threadIdx.x] = *reinterpret_cast<uint32_t*>(&uix);
        }
        else {
            smem[threadIdx.x] = 0;
        }
        __syncwarp();
        uint32_t input_index = base_idx + warp_offset;
        for (int i = 0; i < CUDA_WARP_SIZE; i++) {
            uint32_t s = smem[threadIdx.x - warp_offset + i];
            uint32_t out_idx_me = __popc(s >> (CUDA_WARP_SIZE - warp_offset));
            bool v = (s >> ((CUDA_WARP_SIZE - 1) - warp_offset)) & 0b1;
            if (warp_offset == CUDA_WARP_SIZE - 1) {
                smem_out_idx[warp_index] = out_idx_me + v;
            }
            __syncwarp();
            uint32_t out_idx_warp = smem_out_idx[warp_index];
            if (elements_aquired + out_idx_warp >= CUDA_WARP_SIZE) {
                uint32_t out_idx_me_full = out_idx_me + elements_aquired;
                uint32_t out_idices_idx = warp_base_index + out_idx_me_full;
                if (v && out_idx_me_full < CUDA_WARP_SIZE) {
                    out_indices[out_idices_idx] = input_index;
                }
                __syncwarp();
                output[warp_output_index + warp_offset] =
                    input[out_indices[warp_base_index + warp_offset]];
                __syncwarp();
                if (v && out_idx_me_full >= CUDA_WARP_SIZE) {
                    out_indices[out_idices_idx - CUDA_WARP_SIZE] = input_index;
                }
                elements_aquired += out_idx_warp;
                elements_aquired -= CUDA_WARP_SIZE;
                warp_output_index += CUDA_WARP_SIZE;
            }
            else {
                if (v) {
                    out_indices
                        [warp_base_index + elements_aquired + out_idx_me] =
                            input_index;
                }
                elements_aquired += out_idx_warp;
            }
            input_index += CUDA_WARP_SIZE;
        }
        base_idx += stride;
    }
    __syncwarp();
    if (warp_offset < elements_aquired) {
        output[warp_output_index + warp_offset] =
            input[out_indices[warp_base_index + warp_offset]];
    }
}

struct kernel_3pass_proc_true_striding_optimized_writeout_dispatcher {
    template <typename T, size_t P2>
    static void call(
        T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t* popc,
        size_t chunk_length, size_t element_count, uint32_t chunk_count_p2,
        uint32_t* offset, uint32_t proc_gs)
    {
        kernel_3pass_proc_true_striding_optimized_writeout<P2, T>
            <<<proc_gs, P2>>>(
                input, output, mask, pss, popc, chunk_length, element_count,
                chunk_count_p2, offset);
    }
};

template <class T>
void launch_compact(
    T* d_input, uint8_t* d_mask, T* d_output, size_t element_count)
{
    size_t chunk_length = 1024; // empirical best

    uint32_t* d_pss1;
    uint32_t* d_pss2;
    uint32_t* d_popc;
    uint32_t* d_out_count;
    size_t chunk_count = ceildiv(ceil2mult(element_count, 8), chunk_length);
    size_t intermediate_size_3pass = (chunk_count + 1) * sizeof(uint32_t);
    CUDA_TRY(cudaMalloc(&d_pss1, intermediate_size_3pass));
    CUDA_TRY(cudaMalloc(&d_pss2, intermediate_size_3pass));
    CUDA_TRY(cudaMalloc(&d_popc, intermediate_size_3pass));
    CUDA_TRY(cudaMalloc(&d_out_count, sizeof(uint32_t) * 1));

    // make sure unused bits in bitmask are 0
    int unused_bits = overlap(element_count, 8);
    if (unused_bits) {
        uint8_t* last_mask_byte_ptr = d_mask + element_count / 8;
        uint8_t last_mask_byte = gpu_to_val(last_mask_byte_ptr);
        last_mask_byte >>= unused_bits;
        last_mask_byte <<= unused_bits;
        val_to_gpu(last_mask_byte_ptr, last_mask_byte);
    }
    CUDA_TRY(cudaMemset(d_out_count, 0, 1 * sizeof(*d_out_count)));
    element_count = ceil2mult(
        element_count, 8); // ceil element_count to byte size for kernels

    uint32_t chunk_count_p2 = 1;
    uint32_t max_depth = 0;
    for (; chunk_count_p2 < chunk_count; max_depth++) {
        chunk_count_p2 *= 2;
    }
    uint32_t chunk_length32 = chunk_length / 32;

    uint32_t popc_bs = 32; // empirical, might change
    uint32_t popc_gs = sm_count() * 32;
    if (popc_gs > chunk_count / popc_bs) {
        popc_gs = ceildiv(chunk_count, popc_bs);
    }
    kernel_3pass_popc_none_striding<<<popc_gs, popc_bs>>>(
        d_mask, d_pss1, chunk_length32, element_count);

    cudaMemcpy(
        d_popc, d_pss1, chunk_count * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice);

    uint32_t pss1_bs = 64; // empirical, might change
    // reduce blockcount every depth iteration
    for (int i = 0; i < max_depth; i++) {
        uint32_t blockcount = ((chunk_count >> i) / (pss1_bs * 2)) + 1;
        kernel_3pass_pss_gmem_monolithic<<<blockcount, pss1_bs>>>(
            d_pss1, i, chunk_count, d_out_count);
    }
    // last pass forces result into d_out_count
    kernel_3pass_pss_gmem_monolithic<<<1, 1>>>(
        d_pss1, static_cast<uint8_t>(max_depth), chunk_count, d_out_count);

    uint32_t pss2_bs = 256; // empirical, might change
    uint32_t pss2_gs = (chunk_count / pss2_bs) + 1;
    kernel_3pass_pss2_gmem_monolithic<<<pss2_gs, pss2_bs>>>(
        d_pss1, d_pss2, chunk_count, chunk_count_p2);

    uint32_t proc_bs = 1024;
    uint32_t proc_gs = sm_count() * 32;
    if (proc_gs > chunk_count / proc_bs) {
        proc_gs = ceildiv(chunk_count, proc_bs);
    }
    pow2_template_dispatch<
        kernel_3pass_proc_true_striding_optimized_writeout_dispatcher, 32, 1024,
        T>::
        call(
            proc_bs, d_input, d_output, d_mask, d_pss2, d_popc, chunk_length,
            element_count, chunk_count_p2, (uint32_t*)NULL, proc_gs);

    CUDA_TRY(cudaFree(d_pss1));
    CUDA_TRY(cudaFree(d_pss2));
    CUDA_TRY(cudaFree(d_popc));
    CUDA_TRY(cudaFree(d_out_count));
}
