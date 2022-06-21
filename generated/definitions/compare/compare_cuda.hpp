/*==========================================================================*
 * This file is part of the TVL - a template SIMD library.                  *
 *                                                                          *
 * Copyright 2022 TVL-Team, Database Research Group TU Dresden              *
 *                                                                          *
 * Licensed under the Apache License, Version 2.0 (the "License");          *
 * you may not use this file except in compliance with the License.         *
 * You may obtain a copy of the License at                                  *
 *                                                                          *
 *     http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                          *
 * Unless required by applicable law or agreed to in writing, software      *
 * distributed under the License is distributed on an "AS IS" BASIS,        *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 * See the License for the specific language governing permissions and      *
 * limitations under the License.                                           *
 *==========================================================================*/
/*
 * @file
 * /home/fett/PycharmProjects/TVLGenerator-main/generated/generated/definitions/compare/compare_cuda.hpp
 * @date 20.04.2022
 * @brief Compare primitives. Implementation for cuda
 */

#include "../../declarations/compare.hpp"
#include "compare_cuda_kernel.cuh"

namespace tvl {
namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct equal_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Compares two vector registers for equality.
     * @details todo.
     * @param vec_a Left vector.
     * @param vec_b Right vector.
     * @return Vector mask type indicating whether vec_a[*]==vec_b[*].
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto equal = +[](T a, T b) { return a == b; };
        return launch_elemenwise_compare_op<Vec, equal>(
            vec_a, vec_b, Vec::vector_size_b());
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct less_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Compares two vector registers: a[i] < b[i].
     * @details todo.
     * @param vec_a Left vector.
     * @param vec_b Right vector.
     * @return Vector mask type indicating whether a[i] < b[i]
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto less = +[](T a, T b) { return a < b; };
        return launch_elemenwise_compare_op<Vec, less>(
            vec_a, vec_b, Vec::vector_size_b());
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct lessequal_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Compares two vector registers: a[i] <= b[i].
     * @details todo.
     * @param vec_a Left vector.
     * @param vec_b Right vector.
     * @return Vector mask type indicating whether a[i] <= b[i]
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto lessequal = +[](T a, T b) { return a <= b; };
        return launch_elemenwise_compare_op<Vec, lessequal>(
            vec_a, vec_b, Vec::vector_size_b());
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct greater_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Compares two vector registers: a[i] > b[i].
     * @details todo.
     * @param vec_a Left vector.
     * @param vec_b Right vector.
     * @return Vector mask type indicating whether a[i] > b[i]
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto greater = +[](T a, T b) { return a > b; };
        return launch_elemenwise_compare_op<Vec, greater>(
            vec_a, vec_b, Vec::vector_size_b());
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct greaterequal_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Compares two vector registers: a[i] >= b[i].
     * @details todo.
     * @param vec_a Left vector.
     * @param vec_b Right vector.
     * @return Vector mask type indicating whether a[i] >= b[i]
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto greaterequal = +[](T a, T b) { return a >= b; };
        return launch_elemenwise_compare_op<Vec, greaterequal>(
            vec_a, vec_b, Vec::vector_size_b());
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct between_inclusive_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Checks if the values of a vector are in a specific range (min[*]
     * <= d[*] <= max[*]).
     * @details todo.
     * @param vec_data Data vector.
     * @param vec_min Minimum vector.
     * @param vec_max Maximum vector.
     * @return Vector mask type indicating whether the data is in the given
     * range.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::mask_type apply(
        typename Vec::register_type vec_data,
        typename Vec::register_type vec_min,
        typename Vec::register_type vec_max)
    {
        constexpr auto betweeni =
            +[](T a, T b, T c) { return (a <= b && b <= c); };
        return launch_triplewise_compare_op<Vec, betweeni>(
            vec_min, vec_data, vec_max, Vec::vector_size_b());
    }
};
} // namespace details

} // end of namespace tvl
