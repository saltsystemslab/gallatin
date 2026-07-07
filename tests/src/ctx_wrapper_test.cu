// Exercises the two new features:
//  (1) generate_on_device(..., pinned_per_tree) -- explicit per-block-buffer pinned size.
//  (2) allocator_context<> -- the static-counter context wrapper (per-thread + coalesced).
#include <gallatin/allocators/gallatin.cuh>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
namespace cg = cooperative_groups;
using namespace gallatin::allocators;
using alloc_t = Gallatin<16ULL * 1024 * 1024, 16ULL, 128ULL>;  // 4 trees: 16/32/64/128

__global__ void dump(alloc_t *a) {
  if (threadIdx.x || blockIdx.x) return;
  printf("num_trees=%d\n", a->num_trees);
#if defined(GALLATIN_STATIC_COUNTER) && defined(GALLATIN_BLOCK_CACHE)
  for (int t = 0; t < a->num_trees; t++)
    printf("  tree=%d slice=%lluB g_nblk=%d\n", t,
           (unsigned long long)a->table->get_tree_alloc_size(t), block_cache::S().g_nblk[t]);
#endif
}

// (2a) per-thread context: each thread caches its own slot, allocs, writes tid.
__global__ void ctx_alloc(alloc_t *a, uint64_t n, uint64_t size, uint64_t *misses, uint64_t **ptrs) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  allocator_context<alloc_t> ctx(a, size);
  auto *p = (uint64_t *)ctx.malloc();
  if (!p) { atomicAdd((unsigned long long *)misses, 1ULL); ptrs[tid] = nullptr; return; }
  p[0] = tid; ptrs[tid] = p;
}
__global__ void ctx_verify_free(alloc_t *a, uint64_t n, uint64_t *bad, uint64_t **ptrs) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  uint64_t *p = ptrs[tid];
  if (!p) return;
  if (p[0] != tid) atomicAdd((unsigned long long *)bad, 1ULL);  // aliasing/double-alloc
  allocator_context<alloc_t> ctx(a, 0);
  ctx.free(p);
}

// (2b) coalesced: each tile-16 gets ONE shared allocation via the leader-reserve path.
__global__ void ctx_alloc_tile(alloc_t *a, uint64_t ntiles, uint64_t size, uint64_t *misses, uint64_t **ptrs) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<16>(block);
  uint64_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t tile_id = gtid / 16;
  if (tile_id >= ntiles) return;
  allocator_context<alloc_t> ctx(a, size);
  auto *p = (uint64_t *)ctx.malloc(tile);
  if (tile.thread_rank() == 0) {
    if (!p) atomicAdd((unsigned long long *)misses, 1ULL);
    else { p[0] = tile_id; ptrs[tile_id] = p; }
  }
}

int main(int argc, char **argv) {
  uint64_t pool = 8ULL * 1024 * 1024 * 1024;
  // Explicit PER-TREE pinned sizes for the 4 trees (16/32/64/128). Override each on the
  // cmdline: `ctx_wrapper_test 32 32 32 128`; default keeps the hot 128B tree richly pinned.
  uint32_t pin[4] = {32, 32, 32, 128};
  for (int i = 0; i < 4 && (i + 1) < argc; i++) pin[i] = (uint32_t)strtoul(argv[i + 1], nullptr, 10);
  uint64_t size = 64, n = 4000000;
  printf("=== boot Gallatin<16MB,16,128> with explicit pinned_per_tree={%u,%u,%u,%u} ===\n",
         pin[0], pin[1], pin[2], pin[3]);
  alloc_t *a = alloc_t::generate_on_device(
      pool, 42, /*print_info=*/false, {pin[0], pin[1], pin[2], pin[3]});
  cudaDeviceSynchronize();
  dump<<<1, 1>>>(a); cudaDeviceSynchronize();

  uint64_t *misses, *bad, **ptrs;
  cudaMallocManaged(&misses, 8); cudaMallocManaged(&bad, 8);
  cudaMalloc(&ptrs, sizeof(uint64_t *) * n);
  int TPB = 256;

  *misses = 0; cudaMemset(ptrs, 0, sizeof(uint64_t *) * n);
  ctx_alloc<<<(n + TPB - 1) / TPB, TPB>>>(a, n, size, misses, ptrs); cudaDeviceSynchronize();
  *bad = 0; ctx_verify_free<<<(n + TPB - 1) / TPB, TPB>>>(a, n, bad, ptrs); cudaDeviceSynchronize();
  printf("[per-thread ctx]  size=%llu n=%llu  misses=%llu  back_mismatch=%llu\n",
         (unsigned long long)size, (unsigned long long)n,
         (unsigned long long)*misses, (unsigned long long)*bad);

  uint64_t ntiles = n / 16;
  *misses = 0; cudaMemset(ptrs, 0, sizeof(uint64_t *) * n);
  ctx_alloc_tile<<<(n + TPB - 1) / TPB, TPB>>>(a, ntiles, size, misses, ptrs); cudaDeviceSynchronize();
  cudaError_t e = cudaDeviceSynchronize();
  printf("[coalesced ctx]   tiles=%llu  misses=%llu  cuda=%s\n",
         (unsigned long long)ntiles, (unsigned long long)*misses, cudaGetErrorString(e));
  return 0;
}
