cmake_minimum_required(VERSION 3.15)
project(GEMMULEM C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

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

# Tests
enable_testing()

add_executable(test_em Test/test_em.c)
target_include_directories(test_em PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_em em m)

add_executable(test_distributions Test/test_distributions.c)
target_include_directories(test_distributions PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_distributions em m)

add_executable(test_pearson Test/test_pearson.c)
target_include_directories(test_pearson PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_pearson em m)

add_executable(test_adaptive Test/test_adaptive.c)
target_include_directories(test_adaptive PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_adaptive em m)

add_test(NAME unit_tests COMMAND test_em)
add_test(NAME distribution_tests COMMAND test_distributions)
add_test(NAME pearson_tests COMMAND test_pearson)
add_executable(test_spectral_online_mml Test/test_spectral_online_mml.c)
target_include_directories(test_spectral_online_mml PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_spectral_online_mml em m)

add_executable(test_multivariate Test/test_multivariate.c)
target_include_directories(test_multivariate PRIVATE "${PROJECT_SOURCE_DIR}/src/lib")
target_link_libraries(test_multivariate em m)

add_test(NAME adaptive_tests COMMAND test_adaptive)
add_test(NAME spectral_online_mml_tests COMMAND test_spectral_online_mml)
add_test(NAME multivariate_tests COMMAND test_multivariate)

