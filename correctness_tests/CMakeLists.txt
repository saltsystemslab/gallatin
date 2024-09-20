cmake_minimum_required(VERSION 3.18 FATAL_ERROR)


message("Starting correctness checks")


#setup static library to do CUDA stuff.

file(GLOB_RECURSE static_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/gallatin_static/*.cu)

add_library(gallatin_static STATIC "${static_sources}")

target_include_directories(gallatin_static PUBLIC
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gallatin_static>"
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(gallatin_static PRIVATE gallatin)



include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#add in block test

set(GTEST_GPU_ARCHS "")
evaluate_gpu_archs(GTEST_GPU_ARCHS)

set_target_properties(gallatin_static PROPERTIES CUDA_ARCHITECTURES ${GTEST_GPU_ARCHS})


enable_testing()

include(../cmake/ConfigureTestExecutable.cmake)

ConfigureTestExecutable(error_log_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/error_log_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(block_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/block_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(segment_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/segment_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(veb_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/veb_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(storage_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/storage_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureTestExecutable(poison_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/poison_tests.cc" "${GALLATIN_TESTS_BINARY_DIR}")


# add_executable(
# 	block_tests
# 	"${CMAKE_CURRENT_SOURCE_DIR}/src/block_tests.cc"
# )

# target_link_libraries(
#   block_tests
#   gtest_main
# )

#include final superTest that links everyone

file(GLOB_RECURSE sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_executable(allTests "${sources}")

set_target_properties(allTests PROPERTIES
	CUDA_ARCHITECTURES ${GPU_ARCHS}
    RUNTIME_OUTPUT_DIRECTORY "${GALLATIN_TESTS_BINARY_DIR}"
)

target_link_libraries(allTests PRIVATE gallatin_static)
target_link_libraries(allTests PRIVATE gtest_main)


include(GoogleTest)
gtest_discover_tests(block_tests)