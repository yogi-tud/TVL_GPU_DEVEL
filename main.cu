#include <iostream>
#include "tvlintrin.hpp"
#include <vector>
#include <bitset>
#include "utils.cuh"
#include "cuda_try.cuh"

int main(void)
{

    using namespace tvl;

    using type_t = uint32_t;

    // this sets the tvl extension to cuda instead of sse/avx/avx512
    // look at /extensions/simt/cuda.hpp
    // lets start with 1024 32 bit elements
    using cuda_imp = simd<type_t, cuda, 32768>;

    // generate random data
    size_t ele_count = 32768 / (sizeof(type_t) * 8);
    std::vector<type_t> col = genRandomInts<type_t>(ele_count, 20);
    std::vector<type_t> col2 = genRandomInts<type_t>(ele_count, 10);

    // copy to tvl gpu vector exit
    cuda_imp::register_type vec1 = std::make_shared<cuda_imp::register_type::element_type>(vector_to_gpu(col));
    cuda_imp::register_type vec2 = std::make_shared<cuda_imp::register_type::element_type>(vector_to_gpu(col2));

    // declare resulst vector register
    cuda_imp::register_type vec3;

    // Call TVL add function (look at calc_cuda.hpp)
    vec3 = add<cuda_imp>(vec1, vec2);


    // gpu_buffer_print(vec3, 0, ele_count);

    // Compaction example
    // create bitmask
    std::vector<uint8_t> mask_cpu = create_bitmask(0.1f, 1, ele_count);

    size_t mask_bytes = (ele_count / 8);
    uint8_t* pred = mask_cpu.data();

    cuda_imp::register_type compaction_result;
    cuda_imp::register_type compaction_data = std::make_shared<cuda_imp::register_type::element_type>(vector_to_gpu(col));
    cuda_imp::mask_type compaction_mask = std::make_shared<cuda_imp::mask_type::element_type>();

    CUDA_TRY(cudaMemset(compaction_mask->data, 0, mask_bytes));
    CUDA_TRY(cudaMemcpy(
        compaction_mask->data, &pred[0], mask_bytes, cudaMemcpyHostToDevice));

    // input bitmask
    gpu_buffer_print<uint8_t>(compaction_mask->data, 0, 32);

    // call compaction function
    compaction_result = compaction<cuda_imp>(compaction_data, compaction_mask);

    return 0;
}
