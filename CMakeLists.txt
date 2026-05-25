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


# Check if GPU_ARCHS is predefined
if (DEFINED GPU_ARCHS)
  message("GPU_ARCHS is defined as ${GPU_ARCHS}")
else()
  message("GPU_ARCHS not specified, auto-detecting GPU...")
  set(GPU_ARCHS "")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)

find_package(CUDAToolkit 12.8 REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "12.8")
  message(FATAL_ERROR
    "Gallatin requires CUDA 12.8 or newer (found ${CUDAToolkit_VERSION}). "
    "The atomic publish path uses cuda::atomic_ref<uint16_t>, which was "
    "not supported by libcudacxx until the version that ships with 12.8.")
endif()
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



