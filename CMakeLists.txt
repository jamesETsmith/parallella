cmake_minimum_required(VERSION 3.10)

#
# General Settings
#

# set the project name
project(parallella)
SET(PARALLELLA_MAIN ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS ${PARALLELLA_MAIN})

# User settings
SET(CMAKE_CXX_FLAGS "-g -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Dependencies
#

# External Libraries

SET(EIGEN_ROOT /home/james/apps/eigen/)
SET(CATCH2_ROOT /home/james/Google_Drive/research/projects/parallella/external/Catch2)
SET(JSON_ROOT /home/james/Google_Drive/research/projects/parallella/external/json)

include_directories(${EIGEN_ROOT})
include_directories(${CATCH2_ROOT}/single_include/catch2)
include_directories(${CATCH2_ROOT}/contrib)
add_subdirectory(${CATCH2_ROOT})
include_directories(${JSON_ROOT}/single_include)

# Inlcude parallella

include_directories(./include)

# MKL

list(APPEND CMAKE_MODULE_PATH "${PARALLELLA_MAIN}/cmake")
find_package(MKL REQUIRED)

#
# Compiling
#

# add the test executables
add_executable(test_davidson test/test_davidson.cpp )
add_executable(test_qr test/test_qr.cpp)


if(MKL_FOUND)
    target_link_libraries(test_davidson Catch2::Catch2 ${MKL_LIBRARIES})
    target_link_libraries(test_qr Catch2::Catch2 ${MKL_LIBRARIES})
endif()

#
# Testing
#

list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/contrib")

include(CTest)
enable_testing()
include(ParseAndAddCatchTests)


ParseAndAddCatchTests(test_davidson)
ParseAndAddCatchTests(test_qr)

#
# Benchmark
#

add_executable(benchmark benchmark/bm_davidson.cpp)
