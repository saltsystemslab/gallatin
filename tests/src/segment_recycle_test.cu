// segment_recycle_test
//
// Reproduces the cross-kernel "survivor" pattern from the
// gallatin-segment-recycle-trap bug report:
// /Users/hunter/work/projects/andes_extensions/det_aggregate/andes/docs/issues/
//   gallatin-segment-recycle-trap.md
//
// Structure (matches the bug-report cascade shape):
//
//   Phase A (single kernel, the *cascade* analog):
//      Every thread allocates ONE small "survivor" buffer and writes a
//      tid-marker to it. Then, in the same kernel, it churns through
//      many alloc/free pairs across MULTIPLE tree sizes, so segments
//      fill, drain, and recycle while the survivor sits live. This
//      mirrors aggregate_chain_concat's flow: allocate a buf, then walk
//      chunks freeing many small allocations from the same trees.
//
//   Phase B (separate kernel, the *spawned task* analog):
//      Read each survivor, verify its marker, then global_free it. If
//      the survivor's segment was recycled during Phase A, the marker
//      will be wrong AND/OR the free() will trap at the
//      `tree_id > num_trees` check.
//
// Mixing multiple sizes is the load-bearing detail: a stale malloc from
// tree A whose block was reformatted to tree B is the path that
// historically caused the rollback free_offset to corrupt the wrong
// block's free_counter, recycling a segment that still held a live
// allocation. A single-size workload only triggers within-tree
// reformats, which don't expose the same race.

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace gallatin::allocators;

// Sizes span four adjacent gallatin slice trees
// (default smallest=16, geometric x2: 16, 32, 64, 128 round-to-pow2).
__device__ inline uint64_t size_for(uint64_t x) {
  const uint64_t sizes[4] = {16, 32, 48, 96};
  return sizes[x & 3];
}

__global__ void phase_a_survive_and_churn(uint64_t num_threads, uint32_t churn,
                                          uint64_t **survivors,
                                          uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= num_threads) return;

  // 1. Allocate the survivor (this one outlives the kernel).
  uint64_t survivor_size = size_for(tid);
  uint64_t *survivor = (uint64_t *)global_malloc(survivor_size);
  if (survivor == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    survivors[tid] = nullptr;
    return;
  }
  survivor[0] = tid;
  survivors[tid] = survivor;

  // 2. Churn in the same kernel. Each iteration rotates through sizes
  //    so multiple trees see heavy alloc/free traffic while the
  //    survivor's tree/segment is still loaded.
  for (uint32_t i = 0; i < churn; i++) {
    uint64_t churn_size = size_for(tid + i + 1);
    void *p = global_malloc(churn_size);
    if (p == nullptr) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
      continue;
    }
    *((uint64_t *)p) = tid ^ i;
    global_free(p);
  }
}

__global__ void phase_b_verify_and_free(uint64_t num_threads,
                                        uint64_t **survivors,
                                        uint64_t *corruptions) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= num_threads) return;

  uint64_t *survivor = survivors[tid];
  if (survivor == nullptr) return;

  // Marker check first — if the segment was recycled by a *different*
  // tree, the slice address now lives in foreign allocation space and
  // the marker will be wrong (or wiped to 0 by setup_segment's clear).
  if (survivor[0] != tid) {
    atomicAdd((unsigned long long int *)corruptions, 1ULL);
  }

  // Trap site from the bug report: free() reads chunk_ids[segment] and
  // traps if it's the ~0 sentinel.
  global_free(survivor);
}

int main(int argc, char **argv) {
  uint64_t mem_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
  uint64_t num_threads = 256ULL * 1024;             // 256 K
  uint32_t churn = 256;
  uint32_t iters = 1;  // repeat Phase A this many times before Phase B

  if (argc > 1) num_threads = std::stoull(argv[1]);
  if (argc > 2) churn = static_cast<uint32_t>(std::stoul(argv[2]));
  if (argc > 3) {
    uint64_t gb = std::stoull(argv[3]);
    mem_bytes = gb * 1024ULL * 1024 * 1024;
  }
  if (argc > 4) iters = static_cast<uint32_t>(std::stoul(argv[4]));

  std::cout << "segment_recycle_test:\n"
            << "  num_threads = " << num_threads << "\n"
            << "  churn/thread = " << churn << "\n"
            << "  Phase A iterations = " << iters << "\n"
            << "  allocator mem = " << (mem_bytes >> 30) << " GB\n"
            << "  sizes = {16,32,48,96} (rotate per thread+iter)\n";

  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  uint64_t **survivors;
  GPUErrorCheck(
      cudaMalloc((void **)&survivors, sizeof(uint64_t *) * num_threads));
  GPUErrorCheck(cudaMemset(survivors, 0, sizeof(uint64_t *) * num_threads));

  uint64_t *misses;
  uint64_t *corruptions;
  GPUErrorCheck(cudaMallocManaged((void **)&misses, sizeof(uint64_t)));
  GPUErrorCheck(cudaMallocManaged((void **)&corruptions, sizeof(uint64_t)));
  *misses = 0;
  *corruptions = 0;
  GPUErrorCheck(cudaDeviceSynchronize());

  // The first iteration is the "real" Phase A that produces the
  // survivors. Subsequent iterations (if requested) repeat the same
  // pattern using fresh survivor slots, then we free them before the
  // final pass; this stresses the recycle path more aggressively while
  // keeping memory bounded.
  uint64_t total_ops = 0;
  gallatin::utils::timer t1;
  for (uint32_t it = 0; it < iters; it++) {
    phase_a_survive_and_churn<<<(num_threads - 1) / 256 + 1, 256>>>(
        num_threads, churn, survivors, misses);
    total_ops += uint64_t(num_threads) * (uint64_t(churn) + 1);

    // For all but the last iteration, free survivors and reset the
    // table so the next iteration starts fresh.
    if (it + 1 < iters) {
      phase_b_verify_and_free<<<(num_threads - 1) / 256 + 1, 256>>>(
          num_threads, survivors, corruptions);
      GPUErrorCheck(cudaMemset(survivors, 0, sizeof(uint64_t *) * num_threads));
    }
  }
  double sa = t1.sync_end();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: Phase A: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  std::cout << "  Phase A: " << sa << "s, " << total_ops / sa / 1e9
            << " G ops/s, misses=" << *misses
            << ", inter-iter corruptions=" << *corruptions << "\n";

  if (*corruptions != 0) {
    std::cerr << "FAIL: " << *corruptions
              << " corruptions detected in inter-iteration verify passes\n";
    return 1;
  }

  // Final Phase B: cross-kernel verify+free.
  *corruptions = 0;
  gallatin::utils::timer t2;
  phase_b_verify_and_free<<<(num_threads - 1) / 256 + 1, 256>>>(
      num_threads, survivors, corruptions);
  double sb = t2.sync_end();

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: Phase B: " << cudaGetErrorString(err) << "\n"
              << "  (gallatin-segment-recycle-trap symptom — a live survivor's "
                 "segment\n"
              << "   was recycled between kernels)\n";
    return 1;
  }
  std::cout << "  Phase B (final verify+free): " << sb << "s, "
            << "corruptions=" << *corruptions << "\n";

  int rc = 0;
  if (*corruptions != 0) {
    std::cerr << "FAIL: " << *corruptions
              << " survivor markers were corrupted at cross-kernel boundary\n";
    rc = 1;
  } else {
    std::cout << "PASS\n";
  }

  free_global_allocator();
  cudaFree(survivors);
  cudaFree(misses);
  cudaFree(corruptions);
  return rc;
}
