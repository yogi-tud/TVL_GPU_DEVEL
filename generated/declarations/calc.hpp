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
 * @file /home/fett/PycharmProjects/TVLGenerator-main/generated/generated/declarations/calc.hpp
 * @date 20.04.2022
 * @brief Arithmetic primitives.
 */




namespace tvl {
    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct add_impl{};
    } // end namespace details
    /*
     * @brief Adds two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the addition.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type add(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::add_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct sub_impl{};
    } // end namespace details
    /*
     * @brief element wise substraction of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the subtraction.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type sub(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::sub_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct min_impl{};
    } // end namespace details
    /*
     * @brief element wise minimum of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the minimum.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type min(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::min_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct mul_impl{};
    } // end namespace details
    /*
     * @brief element wise multiplication of two vector registers.
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the multiplication.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type mul(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::mul_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct div_impl{};
    } // end namespace details
    /*
     * @brief element wise division of two vector registers. a[i] / b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the division.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type div(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::div_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct mod_impl{};
    } // end namespace details
    /*
     * @brief element wise modulo of two vector registers. a[i] % b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b Second vector.
     * @return Vector containing result of the modulo operation.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type mod(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::mod_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct shift_left_impl{};
    } // end namespace details
    /*
     * @brief Shift all elements in vec_a p_distance to the left
     * @details todo.
     * @param vec_a First vector.
     * @param p_distance shift distance for all elements
     * @return Vector containing result of the shift operation.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type shift_left(
            typename Vec::register_type  vec_a,
            int  p_distance
    ) {
        return details::shift_left_impl< Vec, Idof >::apply(
                vec_a, p_distance
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct shift_right_impl{};
    } // end namespace details
    /*
     * @brief Shift all elements in vec_a p_distance to the right
     * @details todo.
     * @param vec_a First vector.
     * @param p_distance shift distance for all elements
     * @return Vector containing result of the shift operation.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type shift_right(
            typename Vec::register_type  vec_a,
            int  p_distance
    ) {
        return details::shift_right_impl< Vec, Idof >::apply(
                vec_a, p_distance
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct shift_left_individual_impl{};
    } // end namespace details
    /*
     * @brief Shift all elements in vec_a p_distance to the left. vec_c[i] = vec_a[i] << vec_b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b shift distances.
     * @return Vector containing result of the shift operation.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type shift_left_individual(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::shift_left_individual_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct shift_right_individual_impl{};
    } // end namespace details
    /*
     * @brief Shift all elements in vec_a p_distance to the left. vec_c[i] = vec_a[i] >> vec_b[i]
     * @details todo.
     * @param vec_a First vector.
     * @param vec_b shift distances.
     * @return Vector containing result of the shift operation.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type shift_right_individual(
            typename Vec::register_type  vec_a,
            typename Vec::register_type  vec_b
    ) {
        return details::shift_right_individual_impl< Vec, Idof >::apply(
                vec_a, vec_b
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct invert_impl{};
    } // end namespace details
    /*
     * @brief bitwise invert. 010 ->101
     * @details todo.
     * @param vec_a First vector.
     * @return Vector containing result of the inversion.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type invert(
            typename Vec::register_type  vec_a
    ) {
        return details::invert_impl< Vec, Idof >::apply(
                vec_a
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct hadd_impl{};
    } // end namespace details
    /*
     * @brief horizontal add across a vector register
     * @details todo.
     * @param vec_a First vector.
     * @return Vector containing result of the aggregation at pos[0].
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type hadd(
            typename Vec::register_type  vec_a
    ) {
        return details::hadd_impl< Vec, Idof >::apply(
                vec_a
        );
    }

    namespace details {
        template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof >
        struct compaction_impl{};
    } // end namespace details
    /*
     * @brief compacts a vector register
     * @details todo.
     * @param vec_a First vector.
     * @param mask_a mask.
     * @return Vector containing result of the compaction.
     */
    template< VectorProcessingStyle Vec, ImplementationDegreeOfFreedom Idof = workaround >
    [[nodiscard]]
    TVL_FORCE_INLINE typename Vec::register_type compaction(
            typename Vec::register_type  vec_a,
            typename Vec::mask_type  mask_a
    ) {
        return details::compaction_impl< Vec, Idof >::apply(
                vec_a, mask_a
        );
    }

} // end of namespace tvl

