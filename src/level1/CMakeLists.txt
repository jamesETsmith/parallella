cmake_minimum_required(VERSION 3.10)

# User settings

SET(EIGEN_ROOT /home/james/apps/eigen/)

# set the project name
project(parallella)


# add the executable
add_executable(Tutorial test1.cpp)

include_directories(${EIGEN_ROOT})