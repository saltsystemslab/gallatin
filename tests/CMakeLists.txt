cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



ConfigureExecutable(gallatin_test "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(string_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/string_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(vector_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/vector_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_single_thread "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_single_thread.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_calloc_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_calloc_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/global_churn.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_churn.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_test_noprint "${CMAKE_CURRENT_SOURCE_DIR}/src/global_test_noprint.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(callocable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/callocable_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(callocable_random_test "${CMAKE_CURRENT_SOURCE_DIR}/src/callocable_random_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(chaining_test "${CMAKE_CURRENT_SOURCE_DIR}/src/chaining_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(block_queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/block_queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(coop_chaining_test "${CMAKE_CURRENT_SOURCE_DIR}/src/coop_chaining_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(poison_test "${CMAKE_CURRENT_SOURCE_DIR}/src/poison_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(formattable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/formattable_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(extendible_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/extendible_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(calloc_multiple "${CMAKE_CURRENT_SOURCE_DIR}/src/calloc_multiple.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(ring_queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/ring_queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")


ConfigureExecutable(lambda_enqueue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/lambda_enqueue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(svector_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/svector_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")