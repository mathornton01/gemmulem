cmake_minimum_required(VERSION 3.15)
project(GEMMULEM)

set(CMAKE_CXX_STANDARD 14)

add_library(em
        STATIC
        src/EM.cpp)

add_executable(gemmulem
        src/main.cpp
        )

target_link_libraries(gemmulem em)


# install target
install(TARGETS gemmulem DESTINATION bin)

install(TARGETS em DESTINATION lib)
install(FILES src/EM.h DESTINATION include)

