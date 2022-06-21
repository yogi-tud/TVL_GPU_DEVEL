#include <catch2/catch.hpp>
#include "cuda_test_utils.cuh"

template <
    typename BU,
    bool (*OP)(
        typename BU::cuda_imp::base_type, typename BU::cuda_imp::base_type)>
void validate_elementwise_compare_op_cpu(
    typename BU::buf& a, typename BU::buf& b,
    typename BU::cuda_imp::mask_type res_d)
{
    CUDA_TRY(cudaDeviceSynchronize());
    typedef typename BU::cuda_imp::mask_type::element_type::base_type MT;
    constexpr size_t mt_size_bits = sizeof(MT) * 8;
    auto res_h = gpu_to_vector(
        res_d->data, ceil2mult(a.buf_h.size(), 8 * sizeof(MT)) / (8 * sizeof(MT)));

    // make sure the vector sizes are correct
    REQUIRE(a.buf_h.size() == b.buf_h.size());

    // validate the result contents
    for (size_t me = 0; me < ceildiv(a.buf_h.size(), mt_size_bits);
         me += mt_size_bits) {
        for (int mb = 0; mb < sizeof(MT); mb++) {
            size_t pos = me * mt_size_bits + mb * 8;
            int max = pos + 8 > a.buf_h.size() ? a.buf_h.size() - pos : 8;
            uint8_t mask_byte = *(((uint8_t*)&res_h[me]) + mb);
            for (int i = 0; i < max; i++) {
                REQUIRE(
                    OP(a.buf_h[pos + i], b.buf_h[pos + i]) ==
                    (bool)((mask_byte >> (7 - i)) & 0x1));
            }
        }
    }
}

template <
    typename BU,
    bool (*OP)(
        typename BU::cuda_imp::base_type, typename BU::cuda_imp::base_type,
        typename BU::cuda_imp::base_type)>
void validate_triplewise_compare_op_cpu(
    typename BU::buf& a, typename BU::buf& b, typename BU::buf& c,
    typename BU::cuda_imp::mask_type res_d)
{
    CUDA_TRY(cudaDeviceSynchronize());
    typedef typename BU::cuda_imp::mask_type::element_type::base_type MT;
    constexpr size_t mt_size_bits = sizeof(MT) * 8;
    auto res_h = gpu_to_vector(
        res_d->data, ceil2mult(a.buf_h.size(), 8 * sizeof(MT)) / (8 * sizeof(MT)));

    // make sure the vector sizes are correct
    REQUIRE(a.buf_h.size() == b.buf_h.size());

    // validate the result contents
    for (size_t me = 0; me < a.buf_h.size() / mt_size_bits;
         me += mt_size_bits) {
        for (int mb = 0; mb < sizeof(MT); mb++) {
            size_t pos = me * mt_size_bits + mb * 8;
            int max = pos + 8 > a.buf_h.size() ? a.buf_h.size() - pos : 8;
            uint8_t mask_byte = *(((uint8_t*)&res_h[me]) + mb);
            for (int i = 0; i < max; i++) {
                REQUIRE(
                    OP(a.buf_h[pos + i], b.buf_h[pos + i], c.buf_h[pos + i]) ==
                    (bool)((mask_byte >> (7 - i)) & 0x1));
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("equals", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 3> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::equal<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto equals =
        +[](typename bu::T a, typename bu::T b) { return a == b; };
    validate_elementwise_compare_op_cpu<bu, equals>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("less", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 3> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::less<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto less =
        +[](typename bu::T a, typename bu::T b) { return a < b; };
    validate_elementwise_compare_op_cpu<bu, less>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("lessequal", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 3> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::lessequal<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto lessequal =
        +[](typename bu::T a, typename bu::T b) { return a <= b; };
    validate_elementwise_compare_op_cpu<bu, lessequal>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("greater", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 3> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::greater<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto greater =
        +[](typename bu::T a, typename bu::T b) { return a > b; };
    validate_elementwise_compare_op_cpu<bu, greater>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("greaterequal", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 3> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::greaterequal<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto greaterequal =
        +[](typename bu::T a, typename bu::T b) { return a >= b; };
    validate_elementwise_compare_op_cpu<bu, greaterequal>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("betweeninclusive", "[compare]", cuda_impl_variants)
{
    typedef buffer_util<TestType, 2> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto c = bu::gen_rand_buf();
    auto res_d = tvl::between_inclusive<typename bu::cuda_imp>(
        a.buf_d, b.buf_d, c.buf_d);
    constexpr auto betweeninclusive =
        +[](typename bu::T a, typename bu::T b, typename bu::T c) {
            return (a <= b && b <= c);
        };
    validate_triplewise_compare_op_cpu<bu, betweeninclusive>(b, a, c, res_d);
}
