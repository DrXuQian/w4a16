cmake_minimum_required(VERSION 3.18)
project(fpA_intB_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# SM80 only by default (allow override for PTX JIT).
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

set(FPA_INTB_SOURCES
    cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
    cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fp16_int4_gemm_fg_scalebias.cu
    cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_gemm_sm80_wrappers.cu
    cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu
)

add_library(fpA_intB_gemm STATIC ${FPA_INTB_SOURCES})

set(FPA_INTB_COMPILE_DEFS
    ENABLE_BF16=0
    ENABLE_FP8=0
    ENABLE_FP4=0
)

# Enable Hopper fpA_intB kernels when building for SM90.
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch MATCHES "^90")
        list(APPEND FPA_INTB_COMPILE_DEFS COMPILE_HOPPER_TMA_GEMMS=1)
        break()
    endif()
endforeach()

target_include_directories(fpA_intB_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
)

target_compile_definitions(fpA_intB_gemm PUBLIC ${FPA_INTB_COMPILE_DEFS})

target_compile_options(fpA_intB_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(fpA_intB_gemm
    CUDA::cudart
)

add_executable(test_fpA_intB_gemm test/test_fpA_intB_gemm.cu)

target_link_libraries(test_fpA_intB_gemm
    fpA_intB_gemm
    CUDA::cudart
)

target_compile_definitions(test_fpA_intB_gemm PRIVATE
    ${FPA_INTB_COMPILE_DEFS}
)

target_compile_options(test_fpA_intB_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
