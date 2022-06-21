#include <catch2/catch.hpp>
#include "cuda_test_utils.cuh"

template <
    typename BU,
    typename BU::cuda_imp::base_type (*OP)(
        typename BU::cuda_imp::base_type, typename BU::cuda_imp::base_type)>
void validate_elementwise_op_cpu(
    typename BU::buf& a, typename BU::buf& b,
    typename BU::cuda_imp::register_type res_d)
{
    cudaDeviceSynchronize();
    auto res_h = gpu_to_vector(res_d->data, a.buf_h.size());
    // make sure this is not accidentailly an inplace implementation
    REQUIRE((a.buf_d->data != b.buf_d->data && b.buf_d->data != res_d->data && a.buf_d->data != res_d->data));
    // make sure the vector sizes are correct
    REQUIRE(a.buf_h.size() == b.buf_h.size());
    REQUIRE(a.buf_h.size() == res_h.size());
    for (size_t i = 0; i < a.buf_h.size(); i++) {
        REQUIRE(OP(a.buf_h[i], b.buf_h[i]) == res_h[i]);
    }
}

template <
    typename BU,
    typename BU::cuda_imp::base_type (*OP)(typename BU::cuda_imp::base_type)>
void validate_unary_op_cpu(
    typename BU::buf& a, typename BU::cuda_imp::register_type res_d)
{
    auto res_h = gpu_to_vector(res_d->data, a.buf_h.size());
    // make sure this is not accidentailly an inplace implementation
    REQUIRE(a.buf_d->data != res_d->data);
    REQUIRE(a.buf_h.size() == res_h.size());
    for (size_t i = 0; i < a.buf_h.size(); i++) {
        REQUIRE(OP(a.buf_h[i]) == res_h[i]);
    }
}

TEMPLATE_LIST_TEST_CASE("add", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::add<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto add =
        +[](typename bu::T a, typename bu::T b) { return a + b; };
    validate_elementwise_op_cpu<bu, add>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("sub", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::sub<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto sub =
        +[](typename bu::T a, typename bu::T b) { return a - b; };
    validate_elementwise_op_cpu<bu, sub>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("mul", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::mul<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto mul =
        +[](typename bu::T a, typename bu::T b) { return a * b; };
    validate_elementwise_op_cpu<bu, mul>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("div", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf_non_zero();
    auto res_d = tvl::div<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto div =
        +[](typename bu::T a, typename bu::T b) { return a / b; };
    validate_elementwise_op_cpu<bu, div>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("mod", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf_non_zero();
    auto res_d = tvl::mod<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto mod =
        +[](typename bu::T a, typename bu::T b) { return a % b; };
    validate_elementwise_op_cpu<bu, mod>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("shift_left_individual", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d =
        tvl::shift_left_individual<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto shiftl =
        +[](typename bu::T a, typename bu::T b) { return a << b; };
    validate_elementwise_op_cpu<bu, shiftl>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("shift_right_individual", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d =
        tvl::shift_right_individual<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto shiftr =
        +[](typename bu::T a, typename bu::T b) { return a >> b; };
    validate_elementwise_op_cpu<bu, shiftr>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("min", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto b = bu::gen_rand_buf();
    auto res_d = tvl::min<typename bu::cuda_imp>(a.buf_d, b.buf_d);
    constexpr auto min =
        +[](typename bu::T a, typename bu::T b) { return a < b ? a : b; };
    validate_elementwise_op_cpu<bu, min>(a, b, res_d);
}

TEMPLATE_LIST_TEST_CASE("invert", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto res_d = tvl::invert<typename bu::cuda_imp>(a.buf_d);
    constexpr auto inv = +[](typename bu::T a) { return ~a; };
    validate_unary_op_cpu<bu, inv>(a, res_d);
}
template <typename cuda_imp>
void validate_shift(
    std::vector<typename cuda_imp::base_type> a_h, int shift_val,
    typename cuda_imp::register_type res_d)
{
    CUDA_TRY(cudaDeviceSynchronize());
    auto res_h = gpu_to_vector(res_d->data, a_h.size());
    if (shift_val > 0) {
        std::rotate(a_h.rbegin(), a_h.rbegin() + shift_val, a_h.rend());
    }
    else {
        std::rotate(a_h.begin(), a_h.begin() - shift_val, a_h.end());
    }
    for (size_t i = 0; i < a_h.size(); i++) {
        REQUIRE(a_h[i] == res_h[i]);
    }
}

TEMPLATE_LIST_TEST_CASE("shift", "[calc]", cuda_impl_variants)
{
    typedef buffer_util<TestType> bu;
    auto a = bu::gen_rand_buf();
    auto i = GENERATE(
        0, 1, -1, bu::cuda_imp::vector_element_count() / 2,
        bu::cuda_imp::vector_element_count());

    SECTION("shift_left")
    {
        auto res_d = tvl::shift_left<typename bu::cuda_imp>(a.buf_d, i);
        validate_shift<typename bu::cuda_imp>(a.buf_h, -i, res_d);
    }
    SECTION("shift_right")
    {
        auto res_d = tvl::shift_right<typename bu::cuda_imp>(a.buf_d, i);
        validate_shift<typename bu::cuda_imp>(a.buf_h, i, res_d);
    }
}
