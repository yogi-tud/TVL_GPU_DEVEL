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
 * /home/fett/PycharmProjects/TVLGenerator-main/generated/generated/definitions/calc/calc_cuda.hpp
 * @date 20.04.2022
 * @brief Arithmetic primitives. Implementation for cuda
 */
#ifndef TUD_D2RG_TVL__HOME_FETT_PYCHARMPROJECTS_TVLGENERATOR
#define TUD_D2RG_TVL__HOME_FETT_PYCHARMPROJECTS_TVLGENERATOR

#include "../../declarations/calc.hpp"
#include "calc_cuda_kernel.cuh"

// TODO replace VectorSize/8*sizeof(T) by calls to
// Vec::simd::vector_element_count()

namespace tvl {
namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct add_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Adds two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the addition.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        typename Vec::register_type vec_c;
        size_t element_count = VectorSize / (sizeof(T) * 8);
        // std::cout << "VS in bit: " << VectorSize << std::endl;
        // std::cout << "Ele count: " << element_count << std::endl;
        constexpr auto add = +[](T a, T b) { return a + b; };
        return launch_elemenwise_op<typename Vec::register_type, add>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct sub_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief element wise substraction of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the subtraction.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto sub = +[](T a, T b) { return a - b; };
        return launch_elemenwise_op<typename Vec::register_type, sub>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct min_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief element wise minimum of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the minimum.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto min = +[](T a, T b) { return a < b ? a : b; };
        return launch_elemenwise_op<typename Vec::register_type, min>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct mul_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief element wise multiplication of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the multiplication.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto mul = +[](T a, T b) { return a * b; };
        return launch_elemenwise_op<typename Vec::register_type, mul>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct div_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief element wise division of two vector registers. a[i] / b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the division.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto div = +[](T a, T b) { return a / b; };
        return launch_elemenwise_op<typename Vec::register_type, div>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct mod_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief element wise modulo of two vector registers. a[i] % b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the modulo operation.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto mod = +[](T a, T b) { return a % b; };
        return launch_elemenwise_op<typename Vec::register_type, mod>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct shift_left_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Shift all elements in vec_a p_distance to the left
     * @details todo.
     * @param vec_a First vector.
     * @param p_distance shift distance for all elements
     * @return Vector containing result of the shift operation.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, int p_distance)
    {
        size_t element_count = VectorSize / (sizeof(T) * 8);
        typename Vec::register_type vec_res = std::make_shared<typename Vec::register_type::element_type>();
        launch_dimensions dim = get_launch_dimensions(element_count);
        shift_right_kernel<<<dim.grid_size, dim.block_size>>>(
            vec_a->data, vec_res->data, -p_distance, element_count);
        return vec_res;
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct shift_right_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Shift all elements in vec_a p_distance to the right
     * @details todo.
     * @param vec_a First vector.
     * @param p_distance shift distance for all elements
     * @return Vector containing result of the shift operation.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, int p_distance)
    {
        size_t element_count = VectorSize / (sizeof(T) * 8);
        typename Vec::register_type vec_res = std::make_shared<typename Vec::register_type::element_type>();
        launch_dimensions dim = get_launch_dimensions(element_count);
        shift_right_kernel<<<dim.grid_size, dim.block_size>>>(
            vec_a->data, vec_res->data, p_distance, element_count);
        return vec_res;
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct shift_left_individual_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Shift all elements in vec_a p_distance to the left. vec_c[i] =
     * vec_a[i] << vec_b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b shift distances.
     * @return Vector containing result of the shift operation.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto shiftli = +[](T a, T b) { return a << b; };
        return launch_elemenwise_op<typename Vec::register_type, shiftli>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct shift_right_individual_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief Shift all elements in vec_a p_distance to the right. vec_a[i] =
     * vec_a[i] >> vec_b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b shift distances.
     * @return Vector containing result of the shift operation.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::register_type vec_b)
    {
        constexpr auto shiftri = +[](T a, T b) { return a >> b; };
        return launch_elemenwise_op<typename Vec::register_type, shiftri>(vec_a, vec_b, VectorSize);
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct invert_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief bitwise invert. 010 ->101
     * @details todo.
     * @param vec_a First vector.
     * @return Vector containing result of the inversion.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a)
    {
        size_t element_count = VectorSize / (sizeof(T) * 8);
        launch_dimensions dim = get_launch_dimensions(element_count);
        typename Vec::register_type vec_res = std::make_shared<typename Vec::register_type::element_type>();
        invert_kernel<<<dim.grid_size, dim.block_size>>>(
            vec_a->data, vec_res->data, element_count);
        return vec_res;
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct hadd_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief horizontal add across a vector register
     * @details todo.
     * @param vec_a First vector.
     * @return Vector containing result of the aggregation at pos[0].
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a)
    {
        size_t element_count = VectorSize / (sizeof(T) * 8);
        typename Vec::register_type vec_res = std::make_shared<typename Vec::register_type::element_type>();
        // call cub aggregation here
        //
        launch_dimensions dim = get_launch_dimensions(element_count);
        hadd_kernel<<<dim.grid_size, dim.block_size>>>(
            vec_a->data, vec_res, element_count);
        return vec_res;
    }
};
} // namespace details

namespace details {
template <
    typename T, std::size_t VectorSize, ImplementationDegreeOfFreedom Idof>
struct compaction_impl<simd<T, cuda, VectorSize>, Idof> {
    using Vec = simd<T, cuda, VectorSize>;
    static constexpr bool native_supported()
    {
        return true;
    }
    /*
     * @brief compacts a vector register
     * @details todo.
     * @param vec_a First vector.
     * @param mask_a mask.
     * @return Vector containing result of the compaction.
     */
    [[nodiscard]] TVL_FORCE_INLINE static typename Vec::register_type
    apply(typename Vec::register_type vec_a, typename Vec::mask_type mask_a)
    {
        size_t element_count =
            VectorSize / (sizeof(T) * 8); // element count of vector register
        // std::cout << "VS in bit: " << VectorSize << std::endl;
        // std::cout << "Ele count: " << element_count << std::endl;

        typename Vec::register_type vec_res = std::make_shared<typename Vec::register_type::element_type>();

        // Call compaction here
        launch_compact(vec_a->data, mask_a->data, vec_res->data, element_count);

        return vec_res; // return compaction result
    }
};
} // namespace details

} // end of namespace tvl

#endif // TUD_D2RG_TVL__HOME_FETT_PYCHARMPROJECTS_TVLGENERATOR-MAIN_GENERATED_GENERATED_DEFINITIONS_CALC_CALC_CUDA_HPP
