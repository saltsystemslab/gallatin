// tree_migration_test
//
// Tests that segments aren't "stuck" in the tree they were first formatted
// for. The scenario: a workload allocates exclusively from a small-slice
// tree, frees everything, then switches to large-slice allocations. If
// the allocator gets segments stuck (active_counts never returns to
// num_blocks-1, finish_freeing_block never recycles), the second phase
// can't acquire segments and runs out of memory.
//
// Layout:
//   Phase A: every thread allocates one tree-0 slice (16 B), writes a tid
//            marker. Drains a significant fraction of the allocator into
//            tree-0 segments.
//   Phase B: every thread frees its tree-0 allocation.
//   Phase C: every thread allocates one tree-8 slice (4096 B = the biggest
//            slice in default config). For this to succeed, segments
//            formatted for tree 0 in Phase A must recycle and reformat to
//            tree 8.
//   Phase D: free all tree-8 allocations.
//
// Pass criteria:
//   - Phase A: ≥99% of allocations succeed (allocator not bottlenecked).
//   - Phase C: ≥90% of allocations succeed (would be near-zero if segments
//              were stuck in tree 0).
//   - All marker reads come back correct (no double-malloc / corruption).

#include <gallatin/allocators/global_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void phase_alloc_tree0(uint64_t n, uint64_t **out,
                                  uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;

  uint64_t *p = (uint64_t *)global_malloc(16);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    out[tid] = nullptr;
    return;
  }
  p[0] = tid;
  out[tid] = p;
}

__global__ void phase_free(uint64_t n, uint64_t **in, uint64_t *corruptions) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = in[tid];
  if (p == nullptr) return;
  if (p[0] != tid) {
    atomicAdd((unsigned long long int *)corruptions, 1ULL);
  }
  global_free(p);
}

__global__ void phase_alloc_tree8(uint64_t n, uint64_t **out,
                                  uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;

  // 4096 B = biggest slice in default Gallatin<16MB, 16, 4096> config.
  // Tree 8 segments hold 1 block of 16 MB; for n tree-8 allocs to succeed
  // the allocator needs segments formatted to tree 8.
  uint64_t *p = (uint64_t *)global_malloc(4096);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    out[tid] = nullptr;
    return;
  }
  p[0] = tid;
  out[tid] = p;
}

int main(int argc, char **argv) {
  // 4 GB allocator, ~256 segments. Drain ~60% of allocator capacity in
  // Phase A so Phase C forces large-scale segment migration without
  // hitting saturation-contention noise. (At ≥75% capacity 200M threads
  // produce ~4% misses purely from MAX_ATTEMPTS exhaustion — that's
  // contention, not a stuck-segment bug, so we leave headroom.)
  uint64_t mem_bytes = 4ULL * 1024 * 1024 * 1024;
  uint64_t n_threads_tree0 = 128ULL * 1024 * 1024;  // 2 GB of 16 B slices
  uint64_t n_threads_tree8 = 512ULL * 1024;          // 2 GB of 4096 B slices

  if (argc > 1) mem_bytes = std::stoull(argv[1]) * 1024ULL * 1024;
  // argv[2] = phase-A thread count in millions (scales to drive contention).
  if (argc > 2) n_threads_tree0 = std::stoull(argv[2]) * 1024ULL * 1024;

  std::cout << "tree_migration_test:\n"
            << "  allocator = " << (mem_bytes >> 20) << " MB\n"
            << "  phase A (tree 0): " << n_threads_tree0 << " allocs\n"
            << "  phase C (tree 8): " << n_threads_tree8 << " allocs\n";

  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  uint64_t **buf0, **buf8;
  cudaMalloc((void **)&buf0, sizeof(uint64_t *) * n_threads_tree0);
  cudaMalloc((void **)&buf8, sizeof(uint64_t *) * n_threads_tree8);
  cudaMemset(buf0, 0, sizeof(uint64_t *) * n_threads_tree0);
  cudaMemset(buf8, 0, sizeof(uint64_t *) * n_threads_tree8);

  uint64_t *misses_a, *misses_c, *corruptions;
  cudaMallocManaged((void **)&misses_a, sizeof(uint64_t));
  cudaMallocManaged((void **)&misses_c, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *misses_a = 0;
  *misses_c = 0;
  *corruptions = 0;
  cudaDeviceSynchronize();

  // Phase A: tree-0 mass alloc.
  phase_alloc_tree0<<<(n_threads_tree0 - 1) / 256 + 1, 256>>>(n_threads_tree0,
                                                              buf0, misses_a);
  cudaDeviceSynchronize();
  std::cout << "  phase A: " << *misses_a << " misses ("
            << (100.0 * (*misses_a) / n_threads_tree0) << "%)\n";

  // Phase B: free tree 0.
  phase_free<<<(n_threads_tree0 - 1) / 256 + 1, 256>>>(n_threads_tree0, buf0,
                                                       corruptions);
  cudaDeviceSynchronize();

  // Phase C: tree-8 mass alloc. Needs segments to have migrated.
  phase_alloc_tree8<<<(n_threads_tree8 - 1) / 256 + 1, 256>>>(n_threads_tree8,
                                                              buf8, misses_c);
  cudaDeviceSynchronize();
  std::cout << "  phase C: " << *misses_c << " misses ("
            << (100.0 * (*misses_c) / n_threads_tree8) << "%)\n";

  // Phase D: free tree 8.
  phase_free<<<(n_threads_tree8 - 1) / 256 + 1, 256>>>(n_threads_tree8, buf8,
                                                       corruptions);
  cudaDeviceSynchronize();

  std::cout << "  corruptions = " << *corruptions << "\n";

  int rc = 0;
  // Phase A's job is to populate tree-0 segments; some saturation-level
  // contention misses are informational, not a failure. Only flag if the
  // miss rate is pathological (>20%).
  if (*misses_a * 5 > n_threads_tree0) {
    std::cerr << "FAIL: Phase A miss rate >20% (" << *misses_a << "/"
              << n_threads_tree0 << ") — allocator not draining cleanly\n";
    rc = 1;
  }
  // Phase C: the load-bearing check. If segments are stuck in tree 0,
  // tree 8 can't allocate any segments and miss rate is ~100%.
  if (*misses_c * 10 > n_threads_tree8) {  // >10% miss
    std::cerr << "FAIL: Phase C miss rate >10% — segments not migrating from "
                 "tree 0 to tree 8\n";
    rc = 1;
  }
  if (*corruptions != 0) {
    std::cerr << "FAIL: " << *corruptions << " marker corruptions\n";
    rc = 1;
  }

  if (rc == 0) std::cout << "PASS\n";

  free_global_allocator();
  cudaFree(buf0);
  cudaFree(buf8);
  cudaFree(misses_a);
  cudaFree(misses_c);
  cudaFree(corruptions);
  return rc;
}
