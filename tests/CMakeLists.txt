cmake_minimum_required(VERSION 3.18 FATAL_ERROR)


ConfigureExecutable(gallatin_test "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_test.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/queue_test.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(string_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/string_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(vector_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/vector_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_single_thread "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_single_thread.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(global_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_test.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(global_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/global_churn.cu" "${POGGERS_TESTS_BINARY_DIR}")



ConfigureExecutable(gallatin_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_churn.cu" "${POGGERS_TESTS_BINARY_DIR}")


