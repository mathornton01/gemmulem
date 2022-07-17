add_library(em
        STATIC
        EM.c)

set_property(TARGET em PROPERTY POSITION_INDEPENDENT_CODE ON)
# install target
install(TARGETS em DESTINATION lib)
install(FILES EM.h DESTINATION include)

