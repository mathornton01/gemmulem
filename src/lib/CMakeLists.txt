add_library(em
        STATIC
        vect.c
        EM.c
        distributions.c
        pearson.c
        multivariate.c
        streaming.c
        gpu_estep.c
        simd_estep.c)

set_property(TARGET em PROPERTY POSITION_INDEPENDENT_CODE ON)

# OpenMP for parallel E-step (optional)
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(em PUBLIC OpenMP::OpenMP_C)
    message(STATUS "OpenMP found — parallel E-step enabled")
else()
    message(STATUS "OpenMP not found — single-threaded mode")
endif()

#target_compile_options(em PUBLIC -msse2 -mavx2)

# install target
install(TARGETS em DESTINATION lib)
# Optional OpenCL GPU acceleration (AMD RX 6800 XT and others)
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    target_link_libraries(em PUBLIC OpenCL::OpenCL)
    target_compile_definitions(em PUBLIC GEMMULEM_OPENCL)
    message(STATUS "OpenCL found — GPU E-step enabled (AMD RX 6800 XT, etc.)")
else()
    message(STATUS "OpenCL not found — GPU acceleration disabled (CPU-only mode)")
endif()

# SIMD acceleration: AVX2 for simd_estep.c
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" HAVE_AVX2)
if(HAVE_AVX2)
    set_source_files_properties(simd_estep.c PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -O3")
    message(STATUS "AVX2 available — SIMD E-step enabled (8 doubles/cycle)")
else()
    check_c_compiler_flag("-msse2" HAVE_SSE2)
    if(HAVE_SSE2)
        set_source_files_properties(simd_estep.c PROPERTIES COMPILE_FLAGS "-msse2 -O3")
        message(STATUS "SSE2 available — SIMD E-step enabled (4 doubles/cycle)")
    else()
        message(STATUS "No AVX2/SSE2 — scalar E-step fallback")
    endif()
endif()

install(FILES EM.h distributions.h pearson.h multivariate.h streaming.h gpu_estep.h simd_estep.h DESTINATION include)

