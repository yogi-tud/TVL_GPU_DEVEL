#pragma once

#include <bitset>
#include <vector>
#include <iostream>

#include "cuda_try.cuh"

inline std::vector<uint8_t>
create_bitmask(float selectivity, size_t cluster_count, size_t total_elements)
{
    std::vector<bool> bitset;
    bitset.resize(total_elements);
    size_t total_set_one = selectivity * total_elements;
    size_t cluster_size = total_set_one / cluster_count;
    size_t slice = bitset.size() / cluster_count;

    // start by setting all to zero
    for (int i = 0; i < bitset.size(); i++) {
        bitset[i] = 0;
    }

    for (int i = 0; i < cluster_count; i++) {
        for (int k = 0; k < cluster_size; k++) {
            size_t cluster_offset = i * slice;
            bitset[k + cluster_offset] = 1;
        }
    }

    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements / 8);

    for (int i = 0; i < total_elements / 8; i++) {
        final_bitmask_cpu[i] = 0;
    }

    for (int i = 0; i < bitset.size(); i++) {
        // set bit of uint8
        if (bitset[i]) {
            uint8_t current = final_bitmask_cpu[i / 8];
            int location = i % 8;
            current = 1 << (7 - location);
            uint8_t add_res = final_bitmask_cpu[i / 8];
            add_res = add_res | current;
            final_bitmask_cpu[i / 8] = add_res;
        }
    }

    return final_bitmask_cpu;
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    cudaMemcpy(
        h_buffer, d_buffer + offset, length * sizeof(T),
        cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}
template <typename T> T* vector_to_gpu(const std::vector<T>& vec)
{
    T* buff;
    const auto size = vec.size() * sizeof(T);
    cudaMalloc(&buff, size);
    cudaMemcpy(buff, &vec[0], size, cudaMemcpyHostToDevice);
    return buff;
}

template <typename T> std::vector<T> gpu_to_vector(T* buff, size_t length)
{
    std::vector<T> vec;
    vec.resize(length);
    CUDA_TRY(
        cudaMemcpy(&vec[0], buff, length * sizeof(T), cudaMemcpyDeviceToHost));
    return vec;
}

template <typename T>
static std::vector<T>
genRandomInts(size_t elements, size_t maximum, size_t minimum = 0)
{
    std::vector<T> randoms(elements);
    for (size_t i = 0; i < elements; i++) {
        randoms[i] = rand() % (maximum - minimum) + minimum;
    }

    return randoms;
}
template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (uint32_t i = offset; i < offset + length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

// template arg switcher, e.g. for kernel thread counts
// from 1 to 8192
template <typename DISPATCHER, size_t MIN, size_t MAX, typename... ARGS>
struct pow2_template_dispatch {

    template <
        size_t MIN_MAYBE, size_t MAX_MAYBE, size_t P2, typename... RT_ARGS,
        typename ENABLED =
            typename std::enable_if<(P2 >= MIN_MAYBE && P2 <= MAX_MAYBE)>::type>
    static void call_me_maybe(int dummy, RT_ARGS... rt_args)
    {
        DISPATCHER::template call<ARGS..., P2>(rt_args...);
    }

    template <
        size_t MIN_MAYBE, size_t MAX_MAYBE, size_t P2, typename... RT_ARGS,
        typename ENABLED =
            typename std::enable_if<(P2<MIN_MAYBE || P2> MAX_MAYBE)>::type>
    static void call_me_maybe(float dummy, RT_ARGS... rt_args)
    {
        assert(0);
    }

    template <typename... RT_ARGS> static void call(size_t p2, RT_ARGS... args)
    {
        switch (p2) {
            case 1: call_me_maybe<MIN, MAX, 1>(0, args...); break;
            case 2: call_me_maybe<MIN, MAX, 2>(0, args...); break;
            case 4: call_me_maybe<MIN, MAX, 4>(0, args...); break;
            case 8: call_me_maybe<MIN, MAX, 8>(0, args...); break;
            case 16: call_me_maybe<MIN, MAX, 16>(0, args...); break;
            case 32: call_me_maybe<MIN, MAX, 32>(0, args...); break;
            case 64: call_me_maybe<MIN, MAX, 64>(0, args...); break;
            case 128: call_me_maybe<MIN, MAX, 128>(0, args...); break;
            case 256: call_me_maybe<MIN, MAX, 256>(0, args...); break;
            case 512: call_me_maybe<MIN, MAX, 512>(0, args...); break;
            case 1024: call_me_maybe<MIN, MAX, 1024>(0, args...); break;
            case 2048: call_me_maybe<MIN, MAX, 2048>(0, args...); break;
            case 4096: call_me_maybe<MIN, MAX, 4096>(0, args...); break;
            case 8192: call_me_maybe<MIN, MAX, 8192>(0, args...); break;
            default: assert(0);
        }
    }
};

template <class T> struct dont_deduce_t {
    using type = T;
};

template <typename T>
__device__ __host__ T ceil2mult(T val, typename dont_deduce_t<T>::type mult)
{
    T rem = val % mult;
    if (rem) return val + mult - rem;
    return val;
}

template <typename T>
__device__ __host__ T ceildiv(T div, typename dont_deduce_t<T>::type divisor)
{
    T rem = div / divisor;
    if (rem * divisor == div) return rem;
    return rem + 1;
}

template <typename T>
__device__ __host__ T overlap(T value, typename dont_deduce_t<T>::type align)
{
    T rem = value % align;
    if (rem) return align - rem;
    return 0;
}

template <typename T> T gpu_to_val(T* d_val)
{
    T val;
    CUDA_TRY(cudaMemcpy(&val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
    return val;
}

template <typename T>
void val_to_gpu(T* d_val, typename dont_deduce_t<T>::type val)
{
    CUDA_TRY(cudaMemcpy(d_val, &val, sizeof(T), cudaMemcpyHostToDevice));
}

inline int sm_count()
{
    static int sm_count = []() {
        cudaDeviceProp deviceProp;
        int device;
        CUDA_TRY(cudaGetDevice(&device));
        CUDA_TRY(cudaGetDeviceProperties(&deviceProp, device));
        sm_count = deviceProp.multiProcessorCount;
        return sm_count;
    }(); // for thread safety
    return sm_count;
}

struct launch_dimensions {
    int grid_size;
    int block_size;
};

inline launch_dimensions get_launch_dimensions(size_t vector_size_bits)
{
    return launch_dimensions{256, 1024};
}
