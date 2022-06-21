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
 * @file lib/generated/extensions/simt/cuda.hpp
 * @date 18.03.2022
 * @brief 
 */
#ifndef TUD_D2RG_TVL_LIB_GENERATED_EXTENSIONS_SIMT_CUDA_HPP
#define TUD_D2RG_TVL_LIB_GENERATED_EXTENSIONS_SIMT_CUDA_HPP

//Basically an extension needs a vector register and a mask register
//in case of cuda both are just pointer to gpu memory space
//memory management is NOT part of tvl
//Basetype is just a generic arithmetic datatype (unit8/16/32 etc)
#include <cstddef>
#include <memory>
namespace tvl {
    struct cuda {

        using default_size_in_bits = std::integral_constant< std::size_t, 1024 >;
        template< Arithmetic BaseType, std::size_t VectorSizeInBits = default_size_in_bits::value >
        struct types {

            template <Arithmetic RegisterBaseType, size_t RegisterVectorSizeInBits>
            struct cuda_register {
                typedef RegisterBaseType base_type;
                RegisterBaseType* data;
                cuda_register() {
                    cudaMalloc(&data, VectorSizeInBits);
                }
                cuda_register& operator=(cuda_register& other) {
                    cudaMemcpy(data, other.data, cudaMemcpyDeviceToDevice);
                    return *this;
                }
                cuda_register& operator=(cuda_register&& other) {
                    cudaFree(data);
                    data = other.data;
                    other.data = NULL;
                    return *this;
                }
                cuda_register(cuda_register& other) {
                    cudaMalloc(&data, RegisterVectorSizeInBits);
                    cudaMemcpy(data, other.data, cudaMemcpyDeviceToDevice);
                }
                cuda_register(cuda_register&& other) {
                    data = other.data;
                    other.data = NULL;
                }
                explicit cuda_register(RegisterBaseType* _data):
                    data(_data)
                {}
                ~cuda_register() {
                    cudaFree(data);
                }
            };

            using register_t  =
                std::shared_ptr<cuda_register<BaseType, VectorSizeInBits>>;

            using mask_t =
                std::shared_ptr<cuda_register<uint8_t, (VectorSizeInBits + sizeof(BaseType) - 1) / sizeof(BaseType)>>;
        };
    };
} // end of namespace tvl

#endif //TUD_D2RG_TVL_LIB_GENERATED_EXTENSIONS_SIMT_CUDA_HPP