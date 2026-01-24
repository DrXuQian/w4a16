cmake_minimum_required(VERSION 3.18)
project(marlin_sm110_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(marlin_sm110_kernel STATIC
  marlin_sm110_kernel.cu
  marlin_runner.cu
)
target_include_directories(marlin_sm110_kernel PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(marlin_sm110_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(marlin_sm110_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(test_marlin_sm110 test/test_marlin_sm110.cu)
target_include_directories(test_marlin_sm110 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(test_marlin_sm110 PRIVATE marlin_sm110_kernel CUDA::cudart)
target_compile_options(test_marlin_sm110 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
