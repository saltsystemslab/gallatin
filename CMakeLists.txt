cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gallatin VERSION 1.0 LANGUAGES CXX CUDA)



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

#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)

find_package(CUDAToolkit 11.8 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

#package manager
include(cmake/CPM.cmake)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)

## Package Target

add_library(gallatin INTERFACE)

target_include_directories(gallatin INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(gallatin INTERFACE ssl crypto)


option(LINFO "Add lineInfo for NSight Compute profiling" OFF)

if (LINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(LINFO)


option (GAL_DEBUG "Add debug flags to CMake" OFF)

if (GAL_DEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif(GAL_DEBUG)

option (GAL_DYNAMIC "Add dynamic parallelism - Needed for calloc calls" OFF)

if (GAL_DYNAMIC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudadevrt -rdc=true")
  add_compile_definitions(GALLATIN_USING_DYNAMIC_PARALLELISM=1)
  target_link_libraries(gallatin ${CUDA_cudadevrt_LIBRARY})
  #set_target_properties(gallatin PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)  # this is required for some reason

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif(GAL_DYNAMIC)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
#disable for dcpt
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --time timing.txt -res-usage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 


set_target_properties(gallatin PROPERTIES CMAKE_CUDA_RUNTIME_LIBRARY Static)
set_target_properties(gallatin PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

set(GALLATIN_TESTS_BINARY_DIR "${CMAKE_BINARY_DIR}/tests")

option(GAL_TESTS "Configure CMake to build tests for the allocator" OFF)


option(GAL_DS "Extended testing for auxillary data structures" OFF)

if(GAL_TESTS)
    add_subdirectory(tests)
endif(GAL_TESTS)



