#include <catch2/catch.hpp>
#include "cuda_test_utils.cuh"

template <typename cuda_imp>
void validate_compaction(
    std::vector<typename cuda_imp::base_type> a_h,
    std::vector<typename cuda_imp::mask_type::element_type::base_type> mask_h,
    typename cuda_imp::register_type res_d)
{
    cudaDeviceSynchronize();
    auto res_h = gpu_to_vector(res_d->data, a_h.size());
    size_t output_idx = 0;
    constexpr size_t mask_bits =
        sizeof(
            typename std::remove_pointer<typename cuda_imp::mask_type>::type) *
        8;
    uint8_t* mask = (uint8_t*)&mask_h.front();
    for (size_t i = 0; i < a_h.size(); i++) {
        if (((mask[i / 8] >> (7 - (i % 8))) & 0x1) == 1) {
            REQUIRE(a_h[i] == res_h[output_idx]);
            output_idx++;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("compact", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto m = bu::gen_rand_mask();
    auto res_d = tvl::compaction<typename bu::cuda_imp>(a.buf_d, m.mask_d);
    validate_compaction<typename bu::cuda_imp>(a.buf_h, m.mask_h, res_d);
}
