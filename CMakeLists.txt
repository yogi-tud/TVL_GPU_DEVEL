# convenient cuda clang support was added in 3.19
cmake_minimum_required(VERSION 3.19)
option(USE_CLANG "use clang instead of nvcc" ON)
set(CLANG_VERSION "12" CACHE STRING "specify clang version")

if ("${CLANG_VERSION}" STREQUAL "")
    set(CLANG_VERSION_POSTFIX "")
else()
    set(CLANG_VERSION_POSTFIX "-${CLANG_VERSION}")
endif()
if(${USE_CLANG})
    set(CMAKE_CUDA_COMPILER "clang++${CLANG_VERSION_POSTFIX}")
    set(CMAKE_CXX_COMPILER "clang++${CLANG_VERSION_POSTFIX}")

    # force c++20
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

else()
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# catch 2
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/lib/Catch2")
set(TESTS_INCLUDES
        lib/Catch2/src/
        )
set(TESTS_SOURCES
        tests/calc_cuda.cu
        tests/compare_cuda.cu
        tests/compact_cuda.cu
        tests/tests_main.cpp
        )
add_executable(tests ${TESTS_SOURCES})
add_dependencies(tests Catch2)
target_include_directories(tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(tests Catch2::Catch2)

# don't enable gnu extensions
set(CMAKE_CXX_EXTENSIONS OFF)

project(TVL LANGUAGES CXX CUDA)

add_executable(TVL main.cu)
set_property(TARGET TVL PROPERTY CUDA_ARCHITECTURES 75 61)

set_target_properties(TVL PROPERTIES EXPORT_COMPILE_COMMANDS true)
