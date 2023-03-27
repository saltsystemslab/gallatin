cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(poggers VERSION 0.1.0 LANGUAGES CXX CUDA)



###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)


find_package(CUDAToolkit 11.0 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

#package manager
include(cmake/CPM.cmake)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)

## Package Target

add_library(poggers INTERFACE)

target_include_directories(poggers INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(poggers INTERFACE ssl crypto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 


set_target_properties(poggers PROPERTIES CMAKE_CUDA_RUNTIME_LIBRARY Static)
set_target_properties(poggers PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

set(POGGERS_TESTS_BINARY_DIR "${CMAKE_BINARY_DIR}/tests")

option(POGGERS_BUILD_TESTS "Configure CMake to build tests" OFF)

if(POGGERS_BUILD_TESTS)
    add_subdirectory(tests)
endif(POGGERS_BUILD_TESTS)

