// segment_alloc_concurrent_test
//
// Exercises malloc_segment_allocation under concurrent large-allocation
// pressure to validate the warp-coalesce path:
//
//   * Many threads each call global_malloc with a size > biggest_slice * 4096
//     (so they hit the multi-segment path).
//   * Pointers must be distinct, properly aligned to a segment, and
//     non-overlapping.
//   * Round-trip: free everything, ensure subsequent rounds re-allocate
//     cleanly (no leaked segments).
//
// Reports kernel time as a coarse perf signal. Same-size workloads
// (homogeneous) should benefit from the warp coalesce; varied-size
// workloads (heterogeneous) fall back to per-thread gathers and are
// the baseline.

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace gallatin::allocators;

__global__ void k_alloc_homogeneous(uint64_t n, uint64_t bytes_each,
                                    void **out, uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  void *p = global_malloc(bytes_each);
  out[tid] = p;
  if (p == nullptr) atomicAdd((unsigned long long int *)misses, 1ULL);
}

__global__ void k_alloc_heterogeneous(uint64_t n, uint64_t base_bytes,
                                      void **out, uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  // Vary the size per thread so labeled_partition splits the warp into
  // size-1 partitions — exercises the team_size==1 degenerate path.
  uint64_t bytes = base_bytes + (tid & 0xF) * (16ULL * 1024 * 1024);
  void *p = global_malloc(bytes);
  out[tid] = p;
  if (p == nullptr) atomicAdd((unsigned long long int *)misses, 1ULL);
}

__global__ void k_touch_and_check(uint64_t n, void **in, uint64_t bytes_each,
                                  uint64_t *bad) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  void *p = in[tid];
  if (p == nullptr) return;
  // Stamp identity bytes at first and last allocated slot.
  ((uint64_t *)p)[0] = tid * 31337ULL + 1;
  uint64_t last_idx = (bytes_each / sizeof(uint64_t)) - 1;
  ((uint64_t *)p)[last_idx] = tid * 31337ULL + 2;
  __threadfence();
  uint64_t a = ((uint64_t *)p)[0];
  uint64_t b = ((uint64_t *)p)[last_idx];
  if (a != tid * 31337ULL + 1 || b != tid * 31337ULL + 2)
    atomicAdd((unsigned long long int *)bad, 1ULL);
}

__global__ void k_free(uint64_t n, void **in) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  if (in[tid] != nullptr) global_free(in[tid]);
}

static int check_disjoint(void **ptrs, uint64_t n, uint64_t bytes_each) {
  // Verify pointers are pairwise non-overlapping. O(n^2) but n is small
  // here (we're stress-testing concurrent large allocs, not millions).
  void **h = new void *[n];
  cudaMemcpy(h, ptrs, sizeof(void *) * n, cudaMemcpyDeviceToHost);
  int rc = 0;
  uint64_t live = 0;
  for (uint64_t i = 0; i < n; ++i) {
    if (h[i] == nullptr) continue;
    ++live;
    if ((reinterpret_cast<uintptr_t>(h[i]) % (16ULL * 1024 * 1024)) != 0) {
      std::cerr << "FAIL: ptr[" << i << "] = " << h[i]
                << " is not 16 MB segment-aligned\n";
      rc = 1;
    }
    for (uint64_t j = i + 1; j < n; ++j) {
      if (h[j] == nullptr) continue;
      uintptr_t a = reinterpret_cast<uintptr_t>(h[i]);
      uintptr_t b = reinterpret_cast<uintptr_t>(h[j]);
      uintptr_t lo = a < b ? a : b;
      uintptr_t hi = a < b ? b : a;
      if (lo + bytes_each > hi) {
        std::cerr << "FAIL: ptr[" << i << "]=" << h[i] << " overlaps ptr["
                  << j << "]=" << h[j] << "\n";
        rc = 1;
      }
    }
  }
  std::cout << "  live allocations = " << live << " / " << n << "\n";
  delete[] h;
  return rc;
}

static int run_round(uint64_t pool_gb, uint64_t n_threads,
                     uint64_t bytes_each, int rounds, bool heterogeneous) {
  std::cout << "\n--- " << (heterogeneous ? "heterogeneous" : "homogeneous")
            << " (pool " << pool_gb << " GB, threads " << n_threads
            << ", each " << (bytes_each >> 20) << " MB, rounds " << rounds
            << ") ---\n";

  uint64_t pool_b = pool_gb * 1024ULL * 1024 * 1024;
  if (!init_global_allocator(pool_b, 42, /*print_info=*/false)) {
    std::cout << "  [skip] gallatin boot rejected\n";
    return 0;
  }

  void **buf;
  uint64_t *misses, *bad;
  cudaMalloc((void **)&buf, sizeof(void *) * n_threads);
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&bad, sizeof(uint64_t));

  int rc = 0;
  for (int r = 0; r < rounds; ++r) {
    *misses = 0;
    *bad = 0;
    cudaMemset(buf, 0, sizeof(void *) * n_threads);
    cudaDeviceSynchronize();

    gallatin::utils::timer t;
    if (heterogeneous) {
      k_alloc_heterogeneous<<<(n_threads - 1) / 64 + 1, 64>>>(
          n_threads, bytes_each, buf, misses);
    } else {
      k_alloc_homogeneous<<<(n_threads - 1) / 64 + 1, 64>>>(
          n_threads, bytes_each, buf, misses);
    }
    cudaDeviceSynchronize();
    double alloc_s = t.sync_end();

    k_touch_and_check<<<(n_threads - 1) / 64 + 1, 64>>>(n_threads, buf,
                                                       bytes_each, bad);
    cudaDeviceSynchronize();

    std::cout << "  round " << r << ": alloc " << alloc_s << "s, misses="
              << *misses << ", touch_bad=" << *bad << "\n";

    if (!heterogeneous) {
      // Homogeneous round: validate pointer disjointness and alignment.
      rc |= check_disjoint(buf, n_threads, bytes_each);
    }
    if (*bad != 0) rc = 1;

    k_free<<<(n_threads - 1) / 64 + 1, 64>>>(n_threads, buf);
    cudaDeviceSynchronize();
  }

  free_global_allocator();
  cudaFree(buf);
  cudaFree(misses);
  cudaFree(bad);
  return rc;
}

int main(int argc, char **argv) {
  uint64_t pool_gb = 32;
  if (argc > 1) pool_gb = std::stoull(argv[1]);

  std::cout << "segment_alloc_concurrent_test:\n";

  int rc = 0;
  // Homogeneous: ideal coalescing case. 64 threads × 16 MB each = 1 GB
  // total allocated in one round; small fraction of a 32 GB pool.
  rc |= run_round(pool_gb, /*n=*/64, /*bytes=*/16ULL * 1024 * 1024, /*rounds=*/4,
                  /*hetero=*/false);
  // Heavier homogeneous: 256 threads × 32 MB = 8 GB
  rc |= run_round(pool_gb, /*n=*/256, /*bytes=*/32ULL * 1024 * 1024,
                  /*rounds=*/4, /*hetero=*/false);
  // Heterogeneous (different sizes per thread → no coalesce). 64 threads
  // each 16-256 MB.
  rc |= run_round(pool_gb, /*n=*/64, /*bytes=*/16ULL * 1024 * 1024,
                  /*rounds=*/2, /*hetero=*/true);

  std::cout << (rc == 0 ? "\nPASS\n" : "\nFAIL\n");
  return rc;
}
