add_library(em
        STATIC
        vect.c
        EM.c
        distributions.c
        pearson.c
        multivariate.c
        streaming.c)

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
install(FILES EM.h distributions.h pearson.h multivariate.h streaming.h DESTINATION include)

