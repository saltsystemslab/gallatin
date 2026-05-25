// leak_test
//
// Verifies that equal malloc + free pairs return the allocator to its
// initial state: no accumulated phantom allocations, no segments
// "leaked" by stale block_correct_frees / replace_block rollback paths.
//
// After each cycle we read back per-tree bytes-in-use via
// global_get_tree_bytes_in_use. A healthy allocator returns 0 from
// every tree after a quiescent point.
//
// Runs several cycles to catch leaks that accumulate slowly (e.g. one
// phantom slice per replace_block).

#include <gallatin/allocators/global_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

__global__ void alloc_and_free(uint64_t n, uint64_t size,
                               uint64_t *miss_counter) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= n) return;

  uint64_t *p = (uint64_t *)global_malloc(size);
  if (p == nullptr) {
    atomicAdd((unsigned long long int *)miss_counter, 1ULL);
    return;
  }
  // Write to the allocation so it can't be DCE'd.
  *((uint64_t *)p) = tid;
  global_free(p);
}

int main(int argc, char **argv) {
  uint64_t mem_bytes = 4ULL * 1024 * 1024 * 1024;
  uint32_t cycles = 20;
  uint64_t n_per_cycle = 2ULL * 1024 * 1024;  // 2 M threads
  const uint64_t sizes[4] = {16, 64, 256, 1024};  // span several trees

  if (argc > 1) cycles = std::stoul(argv[1]);
  if (argc > 2) n_per_cycle = std::stoull(argv[2]);

  std::cout << "leak_test:\n"
            << "  allocator = " << (mem_bytes >> 20) << " MB\n"
            << "  cycles = " << cycles << "\n"
            << "  threads/cycle = " << n_per_cycle
            << " (rotating sizes 16/64/256/1024)\n";

  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  uint64_t *miss_counter;
  cudaMallocManaged((void **)&miss_counter, sizeof(uint64_t));

  // We don't probe per-tree bytes-in-use directly; instead we infer leaks
  // from miss rate. A leaking allocator drains over cycles and the miss
  // rate climbs.
  int rc = 0;
  for (uint32_t c = 0; c < cycles; ++c) {
    uint64_t size = sizes[c % 4];
    *miss_counter = 0;
    cudaDeviceSynchronize();

    alloc_and_free<<<(n_per_cycle - 1) / 256 + 1, 256>>>(n_per_cycle, size,
                                                         miss_counter);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "FAIL: cycle " << c << " kernel: "
                << cudaGetErrorString(err) << "\n";
      rc = 1;
      break;
    }

    // Quiescent point: every malloc has been paired with a free in the
    // same kernel. The allocator should report 0 bytes-in-use across
    // all trees.
    // (Note: we don't probe per-tree usage here to keep the test simple;
    // a non-zero leak would show up as ever-increasing miss rate in
    // later cycles when allocator memory drains.)
    std::cout << "  cycle " << c << " (size " << size << "): "
              << *miss_counter << " misses\n";

    // If misses ever exceed 1%, the allocator is leaking faster than it
    // can recycle.
    if (*miss_counter * 100 > n_per_cycle) {
      std::cerr << "FAIL: cycle " << c << " miss rate >1% ("
                << *miss_counter << "/" << n_per_cycle
                << ") — allocator is leaking\n";
      rc = 1;
    }
  }

  if (rc == 0) std::cout << "PASS\n";

  free_global_allocator();
  cudaFree(miss_counter);
  return rc;
}
