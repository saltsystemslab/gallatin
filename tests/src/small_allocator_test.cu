// small_allocator_test
//
// Regression test for the andes_fuzz "boot trap on small allocator" bug:
// when an allocator is sized small enough that there aren't enough segments
// to fully populate the per-tree pinned wavefront (most acutely for the
// largest tree where one slot = one segment), boot used to trap with a
// kernel-launch failure. The fix in boot_block makes that path tolerant —
// slots that can't be filled stay empty, and the malloc fast path probes
// forward to find a populated peer.
//
// This test boots an intentionally small global allocator, exercises a
// handful of allocs across every tree size, and verifies init + alloc +
// free all complete without error.

#include <gallatin/allocators/global_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void smoke_kernel(uint64_t *misses) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= 4096) return;

  // Touch a handful of distinct tree sizes so we hit small-tree slots and
  // large-tree slots both. Mod 9 keeps us inside the default 9-tree config.
  const uint64_t sizes[9] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  uint64_t size = sizes[tid % 9];

  void *p = global_malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)misses, 1ULL);
    return;
  }
  *((uint64_t *)p) = tid;
  global_free(p);
}

int main(int argc, char **argv) {
  // 96 segments × 16 MB = 1.5 GB. This is on the edge — tree-8's wavefront
  // wants MIN_PINNED_CUTOFF=32 segments just for slots, plus segments
  // reserved by smaller trees. The point of the test is exactly that boot
  // should succeed even when there isn't enough for a full wavefront.
  uint64_t mem_bytes = 96ULL * 16ULL * 1024 * 1024;
  if (argc > 1) {
    uint64_t mb = std::stoull(argv[1]);
    mem_bytes = mb * 1024ULL * 1024;
  }

  std::cout << "small_allocator_test:\n"
            << "  allocator mem = " << (mem_bytes >> 20) << " MB ("
            << (mem_bytes / (16ULL * 1024 * 1024)) << " segments)\n";

  // If boot is brittle this call (or the next kernel launch) will
  // throw a CUDA error.
  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: init_global_allocator: " << cudaGetErrorString(err)
              << "\n";
    return 1;
  }

  uint64_t *misses;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  *misses = 0;
  cudaDeviceSynchronize();

  smoke_kernel<<<16, 256>>>(misses);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: smoke_kernel: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  std::cout << "  smoke_kernel done, misses=" << *misses << "\n";

  free_global_allocator();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FAIL: free_global_allocator: " << cudaGetErrorString(err)
              << "\n";
    return 1;
  }

  std::cout << "PASS\n";
  cudaFree(misses);
  return 0;
}
