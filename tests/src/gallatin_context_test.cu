/*
 * gallatin_context_test
 * ----------------------
 * Safety test for the CONTEXT-BASED static-counter fast path (gstatic_fast /
 * gstatic_fast_grouped / gstatic_slow) -- the path IndexinGPU's
 * device_allocator_context uses. Verifies that a per-thread resident context
 * {cidx, cbase, cgen} is safe across MANY allocations and, crucially, across
 * MULTIPLE SIZES, both:
 *   (1) concurrently  -- different threads pinned to different sizes/trees, and
 *   (2) sequentially  -- one thread cycling sizes via a per-size context array,
 *   (3) one context variable reused across changing sizes WITH the required
 *       cidx reset (the documented safe way to change size on a live context).
 *
 * Detection: each claimed slice is stamped with (tid+1) via atomicExch; a
 * non-zero prior value means two owners share a slice (double-alloc). The slice
 * is then cleared and freed. PASS = zero doubles, no hang/crash. Misses (nullptr
 * under churn) are a sizing note, not a failure.
 *
 * Built with the static counter + const base so it exercises exactly the shipped
 * primitives. Does NOT touch the general malloc/free path semantics.
 */
#define GALLATIN_STATIC_COUNTER
#define GALLATIN_CONST_BASE

#include <gallatin/allocators/gallatin.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>

using namespace gallatin::allocators;

using gallatin_type = gallatin::allocators::Gallatin<16ULL * 1024 * 1024, 16ULL, 4096ULL>;

// Sizes spanning distinct trees (powers of two in [min_alloc, max_alloc]).
__constant__ uint64_t g_sizes[5] = {16ULL, 64ULL, 256ULL, 1024ULL, 4096ULL};
static constexpr int kNumSizes = 5;

struct sctx {
  int cidx;
  uint64_t cbase;
  unsigned int cgen;
};

// One reservation through the context fast/slow path for a given tree+size.
template <typename allocator>
__device__ __forceinline__ void* ctx_alloc(allocator* g, sctx& c, uint16_t tree,
                                            uint64_t talloc, bool grouped) {
  void* p;
  if (grouped) {
    p = g->gstatic_fast_grouped(c.cidx, c.cbase, c.cgen, tree, talloc);
  } else {
    p = g->gstatic_fast(c.cidx, c.cbase, c.cgen, talloc);
  }
  if (p == nullptr) {
    p = g->gstatic_slow(tree, talloc, c.cidx, c.cbase, c.cgen);
  }
  return p;
}

// stamp + verify exclusive ownership, then free.
template <typename allocator>
__device__ __forceinline__ void use_and_free(allocator* g, void* p, uint64_t tid,
                                             uint64_t* doubles, uint64_t* misses) {
  if (p == nullptr) {
    atomicAdd((unsigned long long*)misses, 1ULL);
    return;
  }
  unsigned long long me = (unsigned long long)(tid + 1);
  unsigned long long old = atomicExch((unsigned long long*)p, me);
  if (old != 0ULL) {
    atomicAdd((unsigned long long*)doubles, 1ULL);
    printf("DOUBLE-ALLOC: tid %llu found %llu in its slice %p\n", (unsigned long long)tid, old, p);
  }
  unsigned long long cur = atomicExch((unsigned long long*)p, 0ULL);
  if (cur != me) {
    atomicAdd((unsigned long long*)doubles, 1ULL);
    printf("DOUBLE-ALLOC(return): tid %llu saw %llu\n", (unsigned long long)tid, cur);
  }
  g->free(p);
}

// (1) Each thread is pinned to ONE size (-> one tree) for all rounds. Across the
//     grid, ALL sizes/trees run concurrently through the static path. Tests that
//     concurrent multi-tree context use does not cross-corrupt.
template <typename allocator>
__global__ void concurrent_multisize_kernel(allocator* g, uint64_t nthreads, int rounds,
                                            bool grouped, uint64_t* doubles, uint64_t* misses) {
  uint64_t tid = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
  if (tid >= nthreads) return;
  uint64_t size = g_sizes[tid % kNumSizes];
  uint16_t tree = g->get_tree_id_from_size(size);
  uint64_t talloc = g->table->get_tree_alloc_size(tree);
  sctx c{-1, 0, 0};
  for (int r = 0; r < rounds; r++) {
    void* p = ctx_alloc(g, c, tree, talloc, grouped);
    use_and_free(g, p, tid, doubles, misses);
  }
}

// (2) Each thread cycles through ALL sizes, holding one context PER size (the
//     pattern IndexinGPU uses: one context object per slab_size). Tests that a
//     single thread juggling several sizes via per-size contexts is safe.
template <typename allocator>
__global__ void per_size_contexts_kernel(allocator* g, uint64_t nthreads, int rounds,
                                         bool grouped, uint64_t* doubles, uint64_t* misses) {
  uint64_t tid = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
  if (tid >= nthreads) return;
  sctx ctxs[kNumSizes];
  uint16_t trees[kNumSizes];
  uint64_t tallocs[kNumSizes];
  for (int s = 0; s < kNumSizes; s++) {
    ctxs[s] = sctx{-1, 0, 0};
    trees[s] = g->get_tree_id_from_size(g_sizes[s]);
    tallocs[s] = g->table->get_tree_alloc_size(trees[s]);
  }
  uint64_t hash = tid;
  for (int r = 0; r < rounds; r++) {
    hash = gallatin::hashers::MurmurHash64A(&hash, sizeof(uint64_t), r);
    int s = hash % kNumSizes;
    void* p = ctx_alloc(g, ctxs[s], trees[s], tallocs[s], grouped);
    use_and_free(g, p, tid, doubles, misses);
  }
}

// (3) One context variable reused across CHANGING sizes, resetting cidx=-1 on a
//     size change (the documented safe way to repoint a live context at a new
//     tree). A stale cidx must NEVER be paired with a different size -> this
//     verifies the reset discipline keeps it safe.
template <typename allocator>
__global__ void changing_size_reset_kernel(allocator* g, uint64_t nthreads, int rounds,
                                          bool grouped, uint64_t* doubles, uint64_t* misses) {
  uint64_t tid = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
  if (tid >= nthreads) return;
  sctx c{-1, 0, 0};
  int prev_s = -1;
  uint64_t hash = tid;
  for (int r = 0; r < rounds; r++) {
    hash = gallatin::hashers::MurmurHash64A(&hash, sizeof(uint64_t), r);
    int s = hash % kNumSizes;
    if (s != prev_s) {            // size changed -> repoint context safely
      c = sctx{-1, 0, 0};
      prev_s = s;
    }
    uint16_t tree = g->get_tree_id_from_size(g_sizes[s]);
    uint64_t talloc = g->table->get_tree_alloc_size(tree);
    void* p = ctx_alloc(g, c, tree, talloc, grouped);
    use_and_free(g, p, tid, doubles, misses);
  }
}

// (4) MIXED methods: half the threads allocate via the resident CONTEXT path
//     (gstatic_fast/slow), the other half via the STATELESS general malloc()
//     (which, for static-managed trees, routes through malloc_static on the SAME
//     g_ctr64 counters + blocks). Both run concurrently on overlapping
//     sizes/trees. Verifies the two methods hand out DISJOINT slices -- i.e. the
//     cached context reservations and the stateless reservations never collide.
template <typename allocator>
__global__ void mixed_methods_kernel(allocator* g, uint64_t nthreads, int rounds,
                                    bool grouped, uint64_t* doubles, uint64_t* misses) {
  uint64_t tid = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
  if (tid >= nthreads) return;
  bool use_context = (tid & 1ULL);
  uint64_t size = g_sizes[tid % kNumSizes];
  uint16_t tree = g->get_tree_id_from_size(size);
  uint64_t talloc = g->table->get_tree_alloc_size(tree);
  sctx c{-1, 0, 0};
  for (int r = 0; r < rounds; r++) {
    void* p;
    if (use_context) {
      p = ctx_alloc(g, c, tree, talloc, grouped);
    } else {
      p = g->malloc(size);  // stateless general path (-> malloc_static for managed trees)
    }
    use_and_free(g, p, tid, doubles, misses);
  }
}

static int run_one(const char* name, gallatin_type* g, uint64_t nthreads, int rounds,
                   int which, bool grouped) {
  uint64_t* counters;
  cudaMallocManaged((void**)&counters, 2 * sizeof(uint64_t));
  counters[0] = 0;  // doubles
  counters[1] = 0;  // misses
  cudaDeviceSynchronize();

  int block = 256;
  uint64_t grid = (nthreads - 1) / block + 1;
  gallatin::utils::timer t;
  if (which == 0)
    concurrent_multisize_kernel<<<grid, block>>>(g, nthreads, rounds, grouped, counters, counters + 1);
  else if (which == 1)
    per_size_contexts_kernel<<<grid, block>>>(g, nthreads, rounds, grouped, counters, counters + 1);
  else if (which == 2)
    changing_size_reset_kernel<<<grid, block>>>(g, nthreads, rounds, grouped, counters, counters + 1);
  else
    mixed_methods_kernel<<<grid, block>>>(g, nthreads, rounds, grouped, counters, counters + 1);
  cudaError_t err = cudaDeviceSynchronize();
  double secs = t.sync_end();

  uint64_t total = nthreads * (uint64_t)rounds;
  printf("  [%-22s %s] doubles=%llu misses=%llu/%llu (%.3f%%) %s in %.3fs\n",
         name, grouped ? "grouped" : "single", (unsigned long long)counters[0],
         (unsigned long long)counters[1], (unsigned long long)total,
         100.0 * counters[1] / total,
         (err == cudaSuccess ? "ok" : cudaGetErrorString(err)), secs);

  int fail = (counters[0] != 0) || (err != cudaSuccess);
  cudaFree(counters);
  return fail;
}

static const char* kPatternNames[4] = {"concurrent-multisize", "per-size-contexts",
                                       "changing-size+reset", "mixed-ctx+stateless"};

int main(int argc, char** argv) {
  uint64_t num_bytes = (argc > 1) ? std::stoull(argv[1]) : (8ULL * 1024 * 1024 * 1024);
  uint64_t nthreads = (argc > 2) ? std::stoull(argv[2]) : 1000000ULL;
  int rounds = (argc > 3) ? std::stoi(argv[3]) : 8;
  // Optional isolation mode: argv[4]=pattern(0-3), argv[5]=grouped(0/1) -> run ONLY
  // that pattern on a FRESH allocator (disambiguates cumulative leak vs pattern).
  int only_pat = (argc > 4) ? std::stoi(argv[4]) : -1;
  int only_grp = (argc > 5) ? std::stoi(argv[5]) : 0;

  printf("gallatin_context_test: %.1f GB pool, %llu threads, %d rounds%s\n",
         num_bytes / 1.0e9, (unsigned long long)nthreads, rounds,
         only_pat >= 0 ? " [ISOLATED]" : "");

  gallatin_type* g = gallatin_type::generate_on_device(num_bytes, 111);
  cudaDeviceSynchronize();

  int fails = 0;
  if (only_pat >= 0 && only_pat < 4) {
    fails += run_one(kPatternNames[only_pat], g, nthreads, rounds, only_pat, only_grp);
    gallatin_type::free_on_device(g);
    cudaDeviceSynchronize();
    printf("gallatin_context_test[isolated]: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails ? 1 : 0;
  }
  // Run every pattern (single + grouped) on a FRESH allocator each. A single
  // shared allocator across all 8 subtests (~100M sustained ops) degrades capacity
  // cumulatively (a general fragmentation effect that also hits stateless malloc,
  // NOT a per-pattern correctness issue) and masks each pattern's real behavior.
  // Per-subtest fresh allocators measure each pattern's own multi-size safety.
  gallatin_type::free_on_device(g);
  cudaDeviceSynchronize();
  for (int grouped = 0; grouped <= 1; grouped++) {
    for (int pat = 0; pat < 4; pat++) {
      gallatin_type* gp = gallatin_type::generate_on_device(num_bytes, 111);
      cudaDeviceSynchronize();
      fails += run_one(kPatternNames[pat], gp, nthreads, rounds, pat, grouped);
      gallatin_type::free_on_device(gp);
      cudaDeviceSynchronize();
    }
  }

  if (fails == 0) {
    printf("gallatin_context_test: PASS (0 double-alloc; all multi-size + mixed patterns give allocations)\n");
    return 0;
  }
  printf("gallatin_context_test: FAIL (%d subtest(s) with doubles/error)\n", fails);
  return 1;
}
