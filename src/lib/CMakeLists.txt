add_library(em
        STATIC
        vect.c
        EM.c
        distributions.c
        pearson.c)

set_property(TARGET em PROPERTY POSITION_INDEPENDENT_CODE ON)

#target_compile_options(em PUBLIC -msse2 -mavx2)

# install target
install(TARGETS em DESTINATION lib)
install(FILES EM.h distributions.h pearson.h DESTINATION include)

