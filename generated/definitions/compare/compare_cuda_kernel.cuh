#pragma once

#include "../../../utils.cuh"
#include "../../../cuda_try.cuh"

#define CUDA_WARP_SIZE 32
#define ELEMENTS_WARP_STRIDE (CUDA_WARP_SIZE * CUDA_WARP_SIZE)

// kernels
template <typename T, size_t MAX_BLOCK_SIZE, typename MT, bool (*OP)(T, T)>
__global__ void elementwise_compare_op_kernel_big_data(
    T* vec_a, T* vec_b, MT* vec_res, size_t element_count)
{
    static_assert(CUDA_WARP_SIZE % sizeof(MT) == 0);
    constexpr size_t mask_type_bits = sizeof(MT) * 8;
    assert(blockDim.x % CUDA_WARP_SIZE == 0);
    assert(blockDim.x <= MAX_BLOCK_SIZE);
    size_t warps_per_block = blockDim.x / CUDA_WARP_SIZE;
    const size_t grid_stride =
        blockDim.x * gridDim.x * warps_per_block * ELEMENTS_WARP_STRIDE;
    size_t warp_idx = threadIdx.x / CUDA_WARP_SIZE;
    size_t offset_in_warp = threadIdx.x % CUDA_WARP_SIZE;
    __shared__ uint32_t results[MAX_BLOCK_SIZE];
    size_t warp_id_total = warp_idx + blockIdx.x * warps_per_block;
    for (size_t pos = warp_id_total * ELEMENTS_WARP_STRIDE; pos < element_count;
         pos += grid_stride) {
        uint32_t res = 0;
        for (size_t i = 0; i < CUDA_WARP_SIZE; i++) {
            size_t idx = pos + i * CUDA_WARP_SIZE + offset_in_warp;
            bool bit =
                (idx < element_count) ? OP(vec_a[idx], vec_b[idx]) : false;
            res |= bit ? 1 << i : 0;
        }
        results[threadIdx.x] = res;
        __syncwarp();
        uint32_t step_2_res = 0;
        for (size_t i = 0; i < CUDA_WARP_SIZE; i++) {
            uint32_t step_1_res = results[warp_idx * CUDA_WARP_SIZE + i];
            int bit_index = ((i / 8 * 8) + 7 - i % 8);
            step_2_res |= ((step_1_res >> offset_in_warp) & 0x1) << bit_index;
        }
        __syncwarp();
        size_t output_pos = pos / CUDA_WARP_SIZE + offset_in_warp;
        if (output_pos < element_count / CUDA_WARP_SIZE) {
            ((uint32_t*)vec_res)[output_pos] = step_2_res;
        }
        else if (output_pos == element_count / CUDA_WARP_SIZE) {
            for (int i = 0;
                 i < ceildiv(element_count, 8) - output_pos * sizeof(uint32_t);
                 i++) {
                *(((uint8_t*)vec_res) + output_pos * sizeof(uint32_t) + i) =
                    (uint8_t)((step_2_res >> (8 * i)) & 0xFF);
            }
        }
    }
}

template <typename T, typename MT, bool (*OP)(T, T)>
__global__ void elementwise_compare_op_kernel_small_data(
    T* vec_a, T* vec_b, MT* vec_res, size_t element_count)
{
    static_assert(sizeof(MT) < CUDA_WARP_SIZE);
    size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
    bool bit = pos < element_count ? OP(vec_a[pos], vec_b[pos]) : false;

    uint32_t res = __ballot_sync(0xFFFFFFFF, bit);

    if (threadIdx.x <= 3) {
        uint8_t b = (res >> (threadIdx.x * 8)) & 0xFF;
        uint8_t byte_swapped = (b & 0x1) << 7 | (b & 0x2) << 5 |
                               (b & 0x4) << 3 | (b & 0x8) << 1 |
                               (b & 0x10) >> 1 | (b & 0x20) >> 3 |
                               (b & 0x40) >> 5 | (b & 0x80) >> 7;
        ((uint8_t*)vec_res)[pos / 8 + threadIdx.x] = byte_swapped;
    }
}

template <
    typename cuda_impl,
    bool (*OP)(typename cuda_impl::base_type, typename cuda_impl::base_type)>
typename cuda_impl::mask_type launch_elemenwise_compare_op(
    typename cuda_impl::register_type vec_a,
    typename cuda_impl::register_type vec_b, size_t vector_size_bits)
{
    typedef typename cuda_impl::mask_type::element_type::base_type MT;
    typedef typename cuda_impl::base_type T;
    typename cuda_impl::mask_type vec_res = std::make_shared<typename cuda_impl::mask_type::element_type>();
    size_t element_count = vector_size_bits / (sizeof(T) * 8);
    launch_dimensions dim = get_launch_dimensions(element_count);
    if (cuda_impl::vector_element_count() < sm_count() * ELEMENTS_WARP_STRIDE) {
        elementwise_compare_op_kernel_small_data<T, MT, OP>
            <<<ceil2mult(
                   ceildiv(cuda_impl::vector_element_count(), CUDA_WARP_SIZE),
                   ceildiv(sizeof(MT) * 8, CUDA_WARP_SIZE) * CUDA_WARP_SIZE),
               CUDA_WARP_SIZE>>>(vec_a->data, vec_b->data, vec_res->data, element_count);
    }
    else {
        elementwise_compare_op_kernel_big_data<T, 1024, MT, OP>
            <<<dim.grid_size, dim.block_size>>>(
                vec_a->data, vec_b->data, vec_res->data, element_count);
    }
    return vec_res;
}

template <typename T, size_t MAX_BLOCK_SIZE, typename MT, bool (*OP)(T, T, T)>
__global__ void triplewise_compare_op_kernel_big_data(
    T* vec_a, T* vec_b, T* vec_c, MT* vec_res, size_t element_count)
{
    static_assert(CUDA_WARP_SIZE % sizeof(MT) == 0);
    constexpr size_t mask_type_bits = sizeof(MT) * 8;
    assert(blockDim.x % CUDA_WARP_SIZE == 0);
    assert(blockDim.x <= MAX_BLOCK_SIZE);
    size_t warps_per_block = blockDim.x / CUDA_WARP_SIZE;
    const size_t grid_stride =
        blockDim.x * gridDim.x * warps_per_block * ELEMENTS_WARP_STRIDE;
    size_t warp_idx = threadIdx.x / CUDA_WARP_SIZE;
    size_t offset_in_warp = threadIdx.x % CUDA_WARP_SIZE;
    __shared__ uint32_t results[MAX_BLOCK_SIZE];
    size_t warp_id_total = warp_idx + blockIdx.x * warps_per_block;
    for (size_t pos = warp_id_total * ELEMENTS_WARP_STRIDE; pos < element_count;
         pos += grid_stride) {
        uint32_t res = 0;
        for (size_t i = 0; i < CUDA_WARP_SIZE; i++) {
            size_t idx = pos + i * CUDA_WARP_SIZE + offset_in_warp;
            bool bit = (idx < element_count)
                           ? OP(vec_a[idx], vec_b[idx], vec_c[idx])
                           : false;
            res |= bit ? 1 << i : 0;
        }
        results[threadIdx.x] = res;
        __syncwarp();
        uint32_t step_2_res = 0;
        for (size_t i = 0; i < CUDA_WARP_SIZE; i++) {
            uint32_t step_1_res = results[warp_idx * CUDA_WARP_SIZE + i];
            int bit_index = ((i / 8 * 8) + 7 - i % 8);
            step_2_res |= ((step_1_res >> offset_in_warp) & 0x1) << bit_index;
        }
        __syncwarp();
        size_t output_pos = pos / CUDA_WARP_SIZE + offset_in_warp;
        if (output_pos < element_count / CUDA_WARP_SIZE) {
            ((uint32_t*)vec_res)[output_pos] = step_2_res;
        }
        else if (output_pos == element_count / CUDA_WARP_SIZE) {
            for (int i = 0;
                 i < ceildiv(element_count, 8) - output_pos * sizeof(uint32_t);
                 i++) {
                *(((uint8_t*)vec_res) + output_pos * sizeof(uint32_t) + i) =
                    (uint8_t)((step_2_res >> (8 * i)) & 0xFF);
            }
        }
    }
}

template <typename T, typename MT, bool (*OP)(T, T, T)>
__global__ void triplewise_compare_op_kernel_small_data(
    T* vec_a, T* vec_b, T* vec_c, MT* vec_res, size_t element_count)
{
    static_assert(sizeof(MT) < CUDA_WARP_SIZE);
    size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
    bool bit =
        pos < element_count ? OP(vec_a[pos], vec_b[pos], vec_c[pos]) : false;

    uint32_t res = __ballot_sync(0xFFFFFFFF, bit);

    if (threadIdx.x <= 3) {
        uint8_t b = (res >> (threadIdx.x * 8)) & 0xFF;
        uint8_t byte_swapped = (b & 0x1) << 7 | (b & 0x2) << 5 |
                               (b & 0x4) << 3 | (b & 0x8) << 1 |
                               (b & 0x10) >> 1 | (b & 0x20) >> 3 |
                               (b & 0x40) >> 5 | (b & 0x80) >> 7;
        ((uint8_t*)vec_res)[pos / 8 + threadIdx.x] = byte_swapped;
    }
}

template <
    typename cuda_impl,
    bool (*OP)(
        typename cuda_impl::base_type, typename cuda_impl::base_type,
        typename cuda_impl::base_type)>
typename cuda_impl::mask_type launch_triplewise_compare_op(
    typename cuda_impl::register_type vec_a,
    typename cuda_impl::register_type vec_b,
    typename cuda_impl::register_type vec_c, size_t vector_size_bits)
{
    typedef typename cuda_impl::mask_type::element_type::base_type MT;
    typedef typename cuda_impl::base_type T;
    typename cuda_impl::mask_type vec_res = std::make_shared<typename cuda_impl::mask_type::element_type>();
    size_t element_count = vector_size_bits / (sizeof(T) * 8);
    launch_dimensions dim = get_launch_dimensions(element_count);
    if (cuda_impl::vector_element_count() < sm_count() * ELEMENTS_WARP_STRIDE) {
        triplewise_compare_op_kernel_small_data<T, MT, OP>
            <<<ceil2mult(
                   ceildiv(cuda_impl::vector_element_count(), CUDA_WARP_SIZE),
                   ceildiv(sizeof(MT) * 8, CUDA_WARP_SIZE) * CUDA_WARP_SIZE),
               CUDA_WARP_SIZE>>>(vec_a->data, vec_b->data, vec_c->data, vec_res->data, element_count);
    }
    else {
        triplewise_compare_op_kernel_big_data<T, 1024, MT, OP>
            <<<dim.grid_size, dim.block_size>>>(
                vec_a->data, vec_b->data, vec_c->data, vec_res->data, element_count);
    }
    return vec_res;
}
