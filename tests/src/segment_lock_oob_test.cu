// segment_lock_oob_test
//
// Regression test for the OOB write in malloc_segment_allocation.
// tree_locks was allocated with num_trees entries; malloc_segment_allocation
// indexed tree_locks[num_trees] for its global grouping lock — one slot
// past the end. Sanitizer reported "1 byte past 1152-byte tree-lock array"
// on a 100 MB alloc when small-alloc traffic was concurrent.
//
// Repro: spam small concurrent mallocs while one thread issues an alloc
// large enough to take the malloc_segment_allocation path
// (request > bytes_per_segment = 16 MB). Run under
// `compute-sanitizer --tool memcheck` to confirm no OOB.
//
// Pass criterion: kernel completes without sanitizer errors and the big
// allocation returns a non-null pointer.

#include <gallatin/allocators/global_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void background_small_allocs(uint64_t n_iters, uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  // Tight loop hammering the small-alloc path so the per-tree slow paths
  // are active when the big alloc takes the segment lock.
  for (uint64_t i = 0; i < n_iters; ++i) {
    void *p = global_malloc(16 + (tid & 0x3F));
    if (p == nullptr) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
      continue;
    }
    global_free(p);
  }
}

__global__ void big_alloc(uint64_t bytes, void **out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  *out = global_malloc(bytes);
}

__global__ void big_free(void *p) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (p != nullptr) global_free(p);
}

int main(int argc, char **argv) {
  uint64_t mem_bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB allocator
  uint64_t big_bytes = 100ULL * 1024 * 1024;       // 100 MB (matches bug report)
  uint64_t n_bg_threads = 256ULL * 1024;
  uint64_t n_iters = 16;

  if (argc > 1) mem_bytes = std::stoull(argv[1]) * 1024ULL * 1024;

  std::cout << "segment_lock_oob_test:\n"
            << "  allocator = " << (mem_bytes >> 20) << " MB\n"
            << "  big alloc = " << (big_bytes >> 20) << " MB\n"
            << "  bg threads = " << n_bg_threads
            << " x " << n_iters << " iters\n";

  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  uint64_t *misses;
  void **big_ptr;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMallocManaged((void **)&big_ptr, sizeof(void *));
  *misses = 0;
  *big_ptr = nullptr;
  cudaDeviceSynchronize();

  // Run small allocs and the big alloc concurrently on separate streams so
  // gallatin's internal locks see real contention when the big alloc
  // enters malloc_segment_allocation.
  cudaStream_t s_bg, s_big;
  cudaStreamCreate(&s_bg);
  cudaStreamCreate(&s_big);

  background_small_allocs<<<(n_bg_threads - 1) / 256 + 1, 256, 0, s_bg>>>(
      n_iters, misses);
  big_alloc<<<1, 1, 0, s_big>>>(big_bytes, big_ptr);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: kernel error: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  std::cout << "  bg misses = " << *misses << "\n"
            << "  big alloc returned " << (*big_ptr ? "non-null" : "NULL")
            << "\n";

  int rc = 0;
  if (*big_ptr == nullptr) {
    std::cerr << "FAIL: big alloc returned null\n";
    rc = 1;
  }

  big_free<<<1, 1>>>(*big_ptr);
  cudaDeviceSynchronize();

  if (rc == 0) std::cout << "PASS\n";

  free_global_allocator();
  cudaStreamDestroy(s_bg);
  cudaStreamDestroy(s_big);
  cudaFree(misses);
  cudaFree(big_ptr);
  return rc;
}
