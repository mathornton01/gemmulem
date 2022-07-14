add_library(em
        STATIC
        EM.c)

# install target
install(TARGETS em DESTINATION lib)
install(FILES EM.h DESTINATION include)

