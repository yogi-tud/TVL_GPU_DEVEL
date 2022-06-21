#pragma once
#include "../tvlintrin.hpp"

typedef tvl::simd<uint32_t, tvl::cuda, 32> simd_u32_1;
typedef tvl::simd<uint64_t, tvl::cuda, 64> simd_u64_1;
typedef tvl::simd<uint32_t, tvl::cuda, 32 * 7> simd_u32_7;
typedef tvl::simd<uint64_t, tvl::cuda, 64 * 7> simd_u64_7;
typedef tvl::simd<uint32_t, tvl::cuda, 32 * 2047> simd_u32_2047;
typedef tvl::simd<uint64_t, tvl::cuda, 64 * 2047> simd_u64_2047;
typedef tvl::simd<uint64_t, tvl::cuda, 64 * 32769> simd_u64_32769;

// for testing purposes, since multiple cases can get confusing
using cuda_impl_variants_debug =
    std::tuple<tvl::simd<uint64_t, tvl::cuda, 64 * 1025>>;
using cuda_impl_variants = std::tuple<
    simd_u32_1, simd_u64_1, simd_u32_7, simd_u64_7, simd_u32_2047,
    simd_u64_2047, simd_u64_32769>;

template <class CUDA_IMP, size_t RAND_MAX_VAL = 17> class buffer_util {
  public:
    typedef CUDA_IMP cuda_imp;
    typedef typename cuda_imp::base_type T;
    typedef typename cuda_imp::mask_type::element_type::base_type mask_base_type;
    static const T rand_max_val = RAND_MAX_VAL;

  public:
    struct buf {
        typename cuda_imp::register_type buf_d;
        std::vector<T> buf_h;
    };

    struct mask {
        typename cuda_imp::mask_type mask_d;
        std::vector<mask_base_type> mask_h;
    };

    static mask gen_rand_mask()
    {
        mask m;
        constexpr size_t mask_bits = sizeof(mask_base_type) * 8;
        size_t mask_byte_count_padded =
            (cuda_imp::simd::vector_element_count() + (mask_bits - 1)) /
            mask_bits * sizeof(mask_base_type);

        auto mask = create_bitmask(0.5, 1, mask_byte_count_padded * 8);
        m.mask_h = std::vector(
            (typename cuda_imp::mask_type::element_type::base_type*) &mask.front(),
            ((typename cuda_imp::mask_type::element_type::base_type*) &mask.back()) + 1);

        m.mask_d = std::make_shared<typename cuda_imp::mask_type::element_type>(vector_to_gpu(m.mask_h));
        return m;
    }
    static buf gen_rand_buf()
    {
        buf b;
        b.buf_h = genRandomInts<T>(
            cuda_imp::simd::vector_element_count(), rand_max_val);

        b.buf_d = std::make_shared<typename cuda_imp::register_type::element_type>(vector_to_gpu(b.buf_h));
        return b;
    }
    static buf gen_rand_buf_non_zero()
    {
        buf b;
        b.buf_h = genRandomInts<T>(
            cuda_imp::simd::vector_element_count(), rand_max_val, 1);

        b.buf_d = std::make_shared<typename cuda_imp::register_type::element_type>(vector_to_gpu(b.buf_h));
        return b;
    }
};
