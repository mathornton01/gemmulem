cmake_minimum_required(VERSION 3.15)
project(GEMMULEM)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(em
        STATIC
        src/EM.c)

add_executable(gemmulem
        src/main.cpp
        )

target_link_libraries(gemmulem em)


# install target
install(TARGETS gemmulem DESTINATION bin)

install(TARGETS em DESTINATION lib)
install(FILES src/EM.h DESTINATION include)

