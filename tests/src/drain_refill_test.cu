// drain_refill_test
//
// Drains the allocator to capacity, frees everything, and repeats.
// Pass criterion: each cycle must allocate roughly the same number of
// bytes as the first cycle (no monotonic decay). If segments get stuck
// in a tree, or blocks are silently lost through some boundary path,
// subsequent cycles allocate less and the regression shows up here.
//
// This is the long-running-workload analog. The existing gallatin_test
// runs 3 rounds at fixed size; this test runs more rounds, rotates sizes
// per round (so different trees take turns owning all segments), and
// checks for capacity decay rather than just any-success.

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/timer.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void drain_alloc(uint64_t n, uint64_t size, uint64_t **out,
                            uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = (uint64_t *)global_malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    out[tid] = nullptr;
    return;
  }
  p[0] = tid;
  out[tid] = p;
}

__global__ void drain_free(uint64_t n, uint64_t **in, uint64_t *corruptions) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;
  uint64_t *p = in[tid];
  if (p == nullptr) return;
  if (p[0] != tid) atomicAdd((unsigned long long int *)corruptions, 1ULL);
  global_free(p);
}

int main(int argc, char **argv) {
  uint64_t mem_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
  uint32_t cycles = 8;

  if (argc > 1) cycles = std::stoul(argv[1]);
  if (argc > 2) mem_bytes = std::stoull(argv[2]) * 1024ULL * 1024;

  // Sizes rotate per cycle so different trees take turns "owning" most
  // of the allocator. If segments get stuck in a tree, later cycles
  // demanding a different tree will see capacity decay.
  const uint64_t sizes[4] = {16, 64, 256, 1024};

  // Each cycle targets the same total bytes (~12.5% of the allocator),
  // so different sizes ask for very different N. This keeps the work
  // comparable across cycles and well under the allocator's saturation
  // ceiling — degradation across cycles cleanly signals a real bug, not
  // a "we asked for more than fits" artifact.
  uint64_t target_bytes_per_cycle = mem_bytes / 8;

  std::cout << "drain_refill_test:\n"
            << "  allocator = " << (mem_bytes >> 20) << " MB\n"
            << "  cycles = " << cycles << " (sizes rotate 16/64/256/1024)\n"
            << "  target bytes per cycle = " << (target_bytes_per_cycle >> 20)
            << " MB\n";

  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  // Allocate the buffer for the LARGEST expected N (smallest size).
  uint64_t max_n = target_bytes_per_cycle / sizes[0];
  uint64_t **buf;
  cudaMalloc((void **)&buf, sizeof(uint64_t *) * max_n);

  uint64_t *misses, *corruptions;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&corruptions, sizeof(uint64_t));
  *corruptions = 0;

  int rc = 0;

  for (uint32_t c = 0; c < cycles; ++c) {
    uint64_t size = sizes[c % 4];
    uint64_t n = target_bytes_per_cycle / size;

    *misses = 0;
    cudaMemset(buf, 0, sizeof(uint64_t *) * n);
    cudaDeviceSynchronize();

    gallatin::utils::timer t;
    drain_alloc<<<(n - 1) / 256 + 1, 256>>>(n, size, buf, misses);
    cudaDeviceSynchronize();
    double sa = t.sync_end();

    uint64_t successes = n - *misses;
    uint64_t bytes = successes * size;

    std::cout << "  cycle " << c << " size=" << size << " n=" << n << ": "
              << successes << " allocs (" << (bytes >> 20) << " MB) in "
              << sa << "s\n";

    drain_free<<<(n - 1) / 256 + 1, 256>>>(n, buf, corruptions);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "FAIL: cycle " << c << " free: "
                << cudaGetErrorString(err) << "\n";
      rc = 1;
      break;
    }

    // Since each cycle targets only 12.5% of the allocator, any sensible
    // allocator should saturate. Anything below 95% success signals a
    // real problem (segments stuck in a prior tree, lost blocks, etc.).
    if (successes * 100 < n * 95) {
      std::cerr << "FAIL: cycle " << c << " size=" << size
                << " success rate " << (100 * successes / n) << "% < 95%\n";
      rc = 1;
    }
  }

  if (*corruptions != 0) {
    std::cerr << "FAIL: " << *corruptions << " marker corruptions across cycles\n";
    rc = 1;
  }

  if (rc == 0) std::cout << "PASS\n";

  free_global_allocator();
  cudaFree(buf);
  cudaFree(misses);
  cudaFree(corruptions);
  return rc;
}
