cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



#mainline tests

ConfigureExecutable(gallatin_test "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_churn.cu" "${GALLATIN_TESTS_BINARY_DIR}")


ConfigureExecutable(global_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_churn "${CMAKE_CURRENT_SOURCE_DIR}/src/global_churn.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Reproduces the cross-kernel survivor pattern from the
# gallatin-segment-recycle-trap bug report. See test source for details.
ConfigureExecutable(segment_recycle_test "${CMAKE_CURRENT_SOURCE_DIR}/src/segment_recycle_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Regression test for andes_fuzz's gallatin_7c44f77 boot-trap report:
# small allocators must not trap during init even when the per-tree
# wavefront can't be fully populated.
ConfigureExecutable(small_allocator_test "${CMAKE_CURRENT_SOURCE_DIR}/src/small_allocator_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Behavioral / invariants tests: segments must migrate between trees;
# repeated drain/refill must not decay capacity; equal malloc/free pairs
# must not leak.
ConfigureExecutable(tree_migration_test "${CMAKE_CURRENT_SOURCE_DIR}/src/tree_migration_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureExecutable(drain_refill_test "${CMAKE_CURRENT_SOURCE_DIR}/src/drain_refill_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")
ConfigureExecutable(leak_test "${CMAKE_CURRENT_SOURCE_DIR}/src/leak_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Regression for OOB write in malloc_segment_allocation: tree_locks was
# allocated num_trees entries, but the segment-grouping lock indexes
# tree_locks[num_trees]. Run under compute-sanitizer to confirm.
ConfigureExecutable(segment_lock_oob_test "${CMAKE_CURRENT_SOURCE_DIR}/src/segment_lock_oob_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Reproduces the andes report that global_malloc(N) for N between two
# slice-tree sizes returns an undersized slice and causes overlap.
ConfigureExecutable(size_to_tree_overlap_test "${CMAKE_CURRENT_SOURCE_DIR}/src/size_to_tree_overlap_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Probes whether gallatin scales cleanly above 64 GB / 4096 segments.
# Reported regression: 96 GB pool failed where 64 GB worked.
ConfigureExecutable(large_pool_scaling_test "${CMAKE_CURRENT_SOURCE_DIR}/src/large_pool_scaling_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Comprehensive correctness parity test for Gallatin_memory_type::managed
# (UVM-backed allocator) vs device_only. Plus an oversubscription probe.
ConfigureExecutable(managed_mode_test "${CMAKE_CURRENT_SOURCE_DIR}/src/managed_mode_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

# Diagnoses where the ~26% "double_malloc" counter in mixed_churn came
# from by isolating the variables (fixed vs random size, marker scheme).
ConfigureExecutable(churn_diagnostics_test "${CMAKE_CURRENT_SOURCE_DIR}/src/churn_diagnostics_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")


#optional tests
#these are used for development of new features.
# safety and correctness are not guaranteed
if (GAL_DS)


ConfigureExecutable(gallatin_host_test "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_host_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_host_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_host_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")



#ConfigureExecutable(queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(string_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/string_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(vector_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/vector_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_single_thread "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_single_thread.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_test_noprint "${CMAKE_CURRENT_SOURCE_DIR}/src/global_test_noprint.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(callocable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/callocable_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(callocable_random_test "${CMAKE_CURRENT_SOURCE_DIR}/src/callocable_random_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(chaining_test "${CMAKE_CURRENT_SOURCE_DIR}/src/chaining_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(block_queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/block_queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(coop_chaining_test "${CMAKE_CURRENT_SOURCE_DIR}/src/coop_chaining_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(poison_test "${CMAKE_CURRENT_SOURCE_DIR}/src/poison_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(formattable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/formattable_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(even_fill_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/even_fill_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(extendible_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/extendible_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")


#ConfigureExecutable(calloc_multiple "${CMAKE_CURRENT_SOURCE_DIR}/src/calloc_multiple.cu" "${GALLATIN_TESTS_BINARY_DIR}")

#ConfigureExecutable(ring_queue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/ring_queue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")


#ConfigureExecutable(lambda_enqueue_test "${CMAKE_CURRENT_SOURCE_DIR}/src/lambda_enqueue_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(svector_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/svector_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(log_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/log_tests.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(gallatin_managed_test "${CMAKE_CURRENT_SOURCE_DIR}/src/gallatin_managed_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(coop_ext_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/coop_ext_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(bulk_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/bulk_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_exceed_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_exceed_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(global_fused_test "${CMAKE_CURRENT_SOURCE_DIR}/src/global_fused_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(full_ext_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/full_ext_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")

ConfigureExecutable(grow_ext_ht_test "${CMAKE_CURRENT_SOURCE_DIR}/src/grow_ext_ht_test.cu" "${GALLATIN_TESTS_BINARY_DIR}")




endif(GAL_DS)
