// size_to_tree_overlap_test
//
// Reproduce the andes report: a global_malloc(N) for N between two
// slice-tree sizes (e.g., 2048 < 2104 <= 4096) supposedly returns a
// slice from the smaller tree, undersizing the allocation. Subsequent
// allocations then occupy overlapping memory.
//
// Strategy:
//   1. boot gallatin (defaults: 16 B smallest, 4096 B biggest slice)
//   2. malloc 2104 bytes -> ptr1
//   3. malloc 768 bytes  -> ptr2
//   4. print both pointers and the in-allocator tree_id used for each
//   5. FAIL if |ptr2 - ptr1| < 2104 (overlap)
//
// Also probes other "between" sizes (17, 33, 100, 513, 1500, 3000) and
// reports the computed tree_id and slice size — so we can see at a
// glance whether the size->tree mapping rounds up correctly.

#include <gallatin/allocators/global_allocator.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace gallatin::allocators;

struct probe_result {
  uint64_t size;
  uint16_t tree_id;
  uint64_t slice_size;
  void *ptr;
};

__global__ void probe_one(uint64_t size, probe_result *out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint16_t tid = global_gallatin->get_tree_id_from_size(size);
  out->size = size;
  out->tree_id = tid;
  // slice_size = min_size << tree_id; for global default min_size = 16.
  out->slice_size = (tid < global_gallatin->num_trees) ? (16ULL << tid) : 0;
  out->ptr = global_malloc(size);
}

int main() {
  uint64_t mem_bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB
  init_global_allocator(mem_bytes, 42, /*print_info=*/false);

  // The two-allocation overlap test (matches the andes report exactly).
  probe_result *r1, *r2;
  cudaMallocManaged((void **)&r1, sizeof(probe_result));
  cudaMallocManaged((void **)&r2, sizeof(probe_result));

  probe_one<<<1, 1>>>(2104, r1);
  cudaDeviceSynchronize();
  probe_one<<<1, 1>>>(768, r2);
  cudaDeviceSynchronize();

  std::cout << "primary overlap test:\n"
            << "  malloc(2104) -> tree=" << r1->tree_id
            << " slice_size=" << r1->slice_size
            << " ptr=" << r1->ptr << "\n"
            << "  malloc( 768) -> tree=" << r2->tree_id
            << " slice_size=" << r2->slice_size
            << " ptr=" << r2->ptr << "\n";

  int64_t diff = (int64_t)((char *)r2->ptr - (char *)r1->ptr);
  std::cout << "  delta = " << diff << " bytes\n";

  int rc = 0;
  if (diff > 0 && diff < 2104) {
    std::cerr << "FAIL: out_buf overlaps view's requested 2104-byte region\n";
    rc = 1;
  }
  if (diff < 0 && -diff < 768) {
    std::cerr << "FAIL: view overlaps out_buf's requested 768-byte region\n";
    rc = 1;
  }

  // Mapping probe: walk a set of "between" sizes and dump the tree_id
  // each one resolves to. Useful to spot any off-by-one rounding.
  std::cout << "\nsize-to-tree mapping probe:\n";
  const uint64_t sizes[] = {1, 16, 17, 33, 100, 256, 257, 513,
                            1024, 1025, 1500, 2048, 2049, 2104,
                            3000, 4096, 4097};
  probe_result *r;
  cudaMallocManaged((void **)&r, sizeof(probe_result));
  for (uint64_t s : sizes) {
    probe_one<<<1, 1>>>(s, r);
    cudaDeviceSynchronize();
    std::cout << "  size=" << s << " -> tree=" << r->tree_id
              << " slice=" << r->slice_size
              << ((r->slice_size && r->slice_size < s) ? "  *** UNDERSIZED ***"
                                                       : "")
              << "\n";
    if (r->slice_size && r->slice_size < s) rc = 1;
  }

  if (rc == 0) std::cout << "\nPASS\n";

  free_global_allocator();
  cudaFree(r1);
  cudaFree(r2);
  cudaFree(r);
  return rc;
}
