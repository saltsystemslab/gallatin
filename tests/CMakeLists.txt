cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



ConfigureExecutable(test_components "${CMAKE_CURRENT_SOURCE_DIR}/src/test_components.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_key_val_pair "${CMAKE_CURRENT_SOURCE_DIR}/src/test_key_val_pair.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(kmer_speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/kmer_speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(kmer_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/kmer_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(power_2_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/power_2_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

#ConfigureExecutable(separable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/separable_test.cpp" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(free_list_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/free_list_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(bucket_speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_cg_variations "${CMAKE_CURRENT_SOURCE_DIR}/src/test_cg_variations.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_mhm_tcf "${CMAKE_CURRENT_SOURCE_DIR}/src/test_mhm_tcf.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(aligned_stack_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/aligned_stack_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(sub_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/sub_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(cms_init_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/cms_init_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(cms_benchmarks "${CMAKE_CURRENT_SOURCE_DIR}/src/cms_benchmarks.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(cms_variation_benchmarks "${CMAKE_CURRENT_SOURCE_DIR}/src/cms_variation_benchmarks.cu" "${POGGERS_TESTS_BINARY_DIR}")

#ConfigureExecutable(cms_fill_benchmarks "${CMAKE_CURRENT_SOURCE_DIR}/src/cms_fill_benchmarks.cu" "${POGGERS_TESTS_BINARY_DIR}")

#target_compile_options(cms_fill_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#    --relocatable-device-code=true	>)

#set_target_properties(cms_fill_benchmarks PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

ConfigureExecutable(bitarr_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bitarr_benchmark.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(one_thread_bitarr "${CMAKE_CURRENT_SOURCE_DIR}/src/one_thread_bitarr.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(bitarr_comp_verify "${CMAKE_CURRENT_SOURCE_DIR}/src/bitarr_comp_verify.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(buddy_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/buddy_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(templated_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/templated_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(wrapped_templated_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/wrapped_templated_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(four_bitbuddy_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/four_bitbuddy_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(simple_merged_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/simple_merged_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(veb_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/veb_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(one_size_allocator_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/one_size_allocator_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(offset_slab_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/offset_slab_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(beta_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/beta_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(beta_block_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/beta_block_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(delete_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/delete_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(block_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/block_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(sub_veb_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/sub_veb_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(slab_one_size_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/slab_one_size_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(sawtooth_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sawtooth_test.cu" "${POGGERS_TESTS_BINARY_DIR}")


ConfigureExecutable(beta_veb_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/beta_veb_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(beta_slab_one_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/beta_slab_one_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(new_beta_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/new_beta_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")