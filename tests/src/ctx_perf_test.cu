// Throughput comparison for context-based allocation vs the stateless static path.
// All arms do the SAME work: each participating thread performs K malloc+free
// roundtrips of a fixed size. We report Mops = (total mallocs) / elapsed_ms / 1e3.
//
//   A  stateless   a->malloc(size)            (warp-coalesced internally: 1 atomic/warp)
//   B  ctx thread  ctx.malloc()               (cached slot: 1 atomic/thread)
//   C  ctx tile16  ctx.malloc(tile16)         (leader reserve: 1 atomic/tile)
//   D  ctx tile32  ctx.malloc(tile32)         (leader reserve: 1 atomic/warp)
#include <gallatin/allocators/gallatin.cuh>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
namespace cg = cooperative_groups;
using namespace gallatin::allocators;
using alloc_t = Gallatin<16ULL * 1024 * 1024, 16ULL, 128ULL>;

// A: stateless per-thread (malloc_static coalesces the warp).
__global__ void bench_stateless(alloc_t *a, uint64_t nthreads, uint64_t size,
                                uint64_t K, uint64_t *miss) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nthreads) return;
  uint64_t local_miss = 0;
  for (uint64_t k = 0; k < K; k++) {
    void *p = a->malloc(size);
    if (!p) { local_miss++; continue; }
    ((volatile uint64_t *)p)[0] = tid;
    a->free(p);
  }
  if (local_miss) atomicAdd((unsigned long long *)miss, local_miss);
}

// B: per-thread context, constructed once, reused across the loop.
__global__ void bench_ctx_thread(alloc_t *a, uint64_t nthreads, uint64_t size,
                                 uint64_t K, uint64_t *miss) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nthreads) return;
  allocator_context<alloc_t> ctx(a, size);
  uint64_t local_miss = 0;
  for (uint64_t k = 0; k < K; k++) {
    void *p = ctx.malloc();
    if (!p) { local_miss++; continue; }
    ((volatile uint64_t *)p)[0] = tid;
    ctx.free(p);
  }
  if (local_miss) atomicAdd((unsigned long long *)miss, local_miss);
}

// C/D: coalesced context; TILE = 16 or 32. Only the leader holds the allocation.
template <int TILE>
__global__ void bench_ctx_tile(alloc_t *a, uint64_t ntiles, uint64_t size,
                               uint64_t K, uint64_t *miss) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE>(block);
  uint64_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t tile_id = gtid / TILE;
  if (tile_id >= ntiles) return;
  allocator_context<alloc_t> ctx(a, size);
  uint64_t local_miss = 0;
  for (uint64_t k = 0; k < K; k++) {
    void *p = ctx.malloc(tile);  // one shared slice, broadcast to every lane
    if (!p) { if (tile.thread_rank() == 0) local_miss++; continue; }
    if (tile.thread_rank() == 0) {
      ((volatile uint64_t *)p)[0] = tile_id;
      ctx.free(p);  // exactly ONE free per tile (all lanes hold the same p)
    }
    tile.sync();
  }
  if (tile.thread_rank() == 0 && local_miss)
    atomicAdd((unsigned long long *)miss, local_miss);
}

int main(int argc, char **argv) {
  uint64_t pool = 8ULL * 1024 * 1024 * 1024;
  uint64_t size = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 64;
  uint64_t nthreads = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ULL << 20);  // 1M
  uint64_t K = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 64;
  // Pin all four trees generously so buffer capacity isn't the bottleneck.
  alloc_t *a = alloc_t::generate_on_device(pool, 42, false, {256, 256, 256, 256});
  cudaDeviceSynchronize();

  uint64_t *miss; cudaMallocManaged(&miss, 8);
  int TPB = 256;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  float ms; double total = (double)nthreads * K;

  auto run = [&](const char *name, auto launch) {
    *miss = 0;
    launch();  // warmup
    cudaDeviceSynchronize();
    *miss = 0;
    cudaEventRecord(s);
    launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    cudaError_t err = cudaGetLastError();
    printf("%-14s size=%llu ops=%.0f  %8.3f ms  %8.2f Mops  miss=%llu  %s\n",
           name, (unsigned long long)size, total, ms, total / (ms * 1e3),
           (unsigned long long)*miss, cudaGetErrorString(err));
  };

  run("A stateless", [&] {
    bench_stateless<<<(nthreads + TPB - 1) / TPB, TPB>>>(a, nthreads, size, K, miss);
  });
  run("B ctx-thread", [&] {
    bench_ctx_thread<<<(nthreads + TPB - 1) / TPB, TPB>>>(a, nthreads, size, K, miss);
  });
  {
    uint64_t ntiles = nthreads / 16; total = (double)ntiles * K;
    run("C ctx-tile16", [&] {
      bench_ctx_tile<16><<<(nthreads + TPB - 1) / TPB, TPB>>>(a, ntiles, size, K, miss);
    });
  }
  {
    uint64_t ntiles = nthreads / 32; total = (double)ntiles * K;
    run("D ctx-tile32", [&] {
      bench_ctx_tile<32><<<(nthreads + TPB - 1) / TPB, TPB>>>(a, ntiles, size, K, miss);
    });
  }
  return 0;
}
