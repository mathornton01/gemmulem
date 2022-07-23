cmake_minimum_required(VERSION 3.15)
project(GEMMULEM)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/lib)

add_executable(gemmulem
        src/main.cpp
        )

target_include_directories(gemmulem PUBLIC
        "${PROJECT_SOURCE_DIR}/src/lib"
        )

target_link_libraries(gemmulem em)

# install target
install(TARGETS gemmulem DESTINATION bin)

