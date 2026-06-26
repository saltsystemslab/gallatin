#ifndef GALLATIN_ALLOCATOR
#define GALLATIN_ALLOCATOR
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
//  portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.


/*** ABOUT
  Gallatin is a generic vEB-based GPU allocator that allows for individual
threads on the GPU to perform allocations.

When building the type, the template arguments are as follows:

* bytes_per_segment: Bytes per segment. Defualt 16 megabytes
  - This is the granularity that GPU memory is sliced into.
* uint64_t smallest: Number of bytes in the smallest slice size. Defualt 16 bytes
* uint64_t biggest: Number of bytes in the largest slice size. Defualt 4 kilobytes

Based on these template parameters, the number of trees
 and intermediate slice sizes are determined at compile-time.


Usage: 
  Gallatin must be constructed and destructed by host.
To do so, call Gallatin<template_args>::generate_on_device()
and supply the # of bytes to be made allocable, along with a random seed.
This function returns a handle to the allocator that can be used in device kernels.

To free device memory at the end of execution, call

 Gallatin<template_args>::free_on_device(your_pointer);

This will free the associated device memory, including all memory that has been handed out.
THIS WILL NOT WIPE DEVICE POINTERS. 
Using memory allocated by Gallatin after this call is undefined behavior.

Inside of a kernel, you must pass a pointer to the allocator.
You can then allocate new memory with the malloc method:

  void * alloc_ptr->malloc(uint64_t num_bytes)

This returns a void * type of at least num_bytes(), or nullptr if no allocation is available.

Once the memory is no longer needed, it can be returned via

  void alloc_ptr->free(void * memory_ptr);

The pointer returned must be the same address that was returned - 
  trying to free a different address can result in undefined behavior.

*/




// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin/allocators/memory_table.cuh>
#include <gallatin/allocators/shared_block_storage.cuh>
#include <gallatin/allocators/murmurhash.cuh>

#define GALLATIN_TRAP_ON_ERR 1

#ifndef GALLATIN_DEBUG_PRINTS
#define GALLATIN_DEBUG_PRINTS 0
#endif

namespace gallatin {

namespace allocators {


//Change these to set # of times Gallatin attempts allocation ops.
//Lowering these ups the chance of weak memory issues,
//meaning that Gallatin can fail to allocate even when memory is available

//However, lowering these values does reduce the time taken to ascertain
//that no allocation is available, which could be advantageous in some niche scenario
//when you know you will perform more allocations than is possible.

//Consequently, these are left as modifiable values
//Correctness is only guaranteed at the set values, change at your own risk.
#define REQUEST_BLOCK_MAX_ATTEMPTS 1000
#define GALLATIN_MAX_ATTEMPTS 500
#define GALLATIN_MALLOC_LOOP_ATTEMPTS 10
#define GALLATIN_MALLOC_BLOCK_ATTEMPTS 500
#define GALLATIN_MALLOC_SEGMENT_ATTEMPTS 500

#define GALLATIN_BLOCK_CHECK 0

//Macros for controlling system behavior
//Reregister cutoff determines the % of fill at which the allocator
//adds exhausted segments back to their tree.

// MIN_PINNED_CUTOFF is the minimum number of live blocks in the per-tree
// wavefront. Each pinned slot is keyed by (smid ^ warp_in_block ^ blockIdx).
// Bumping the cutoff higher in principle reduces the per-slot collision
// factor, but it also makes the boot kernel acquire `cutoff * num_trees`
// blocks from segment_tree before any user kernel runs — and the largest
// trees yield only one block per segment, so a high cutoff can saturate
// the segment budget at startup. 32 is the empirical sweet spot for now.

//Team free controls if opportunistic coalescing is used for frees
#define REREGISTER_CUTOFF .1
#ifndef MIN_PINNED_CUTOFF
#define MIN_PINNED_CUTOFF 32
#endif
#define GALLATIN_TEAM_FREE 1


// alloc table associates chunks of memory with trees

// using uint16_t as there shouldn't be that many trees.

// register atomically inserst tree num, or registers memory from segment_tree.

using namespace gallatin::utils;

static __global__ void boot_segment_trees(veb_tree **segment_trees,
                                   uint64_t max_chunks, int num_trees) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= max_chunks) return;

  for (int i = 0; i < num_trees; i++) {
    segment_trees[i]->remove(tid);
  }
}

// Deterministic boot helper: like boot_segment_trees but also pre-claims
// the boot range from segment_tree and pre-publishes those segments into
// their owning sub_tree. This eliminates the runtime contention on
// segment_tree->malloc_first() during boot_shared_block_container, which
// is what makes compute-sanitizer livelock with MIN_PINNED_CUTOFF=32 on
// Blackwell — too many concurrent threads spinning in successor_thorough.
//
// `boot_segment_offsets` is a uint64_t array of size num_trees+1. Tree t
// owns segments [boot_segment_offsets[t], boot_segment_offsets[t+1]).
// total_boot_segments == boot_segment_offsets[num_trees].
static __global__ void boot_segments_deterministic(
    veb_tree *segment_tree,
    veb_tree **sub_trees,
    uint64_t max_chunks,
    int num_trees,
    uint64_t total_boot_segments,
    const uint64_t *boot_segment_offsets) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= max_chunks) return;

  // Clear every sub_tree bit at this index (sub_trees start full per
  // generate_on_device_nowait). Mirrors the old boot_segment_trees loop.
  for (int i = 0; i < num_trees; i++) {
    sub_trees[i]->remove(tid);
  }

  // Pre-claim boot segments from segment_tree and publish them into the
  // owning tree's sub_tree. Single thread per segment, deterministic
  // addresses — no contention beyond the natural per-64bit-word atomic.
  if (tid < total_boot_segments) {
    segment_tree->remove(tid);

    // Find owning tree by walking the offset table. num_trees is tiny
    // (typically 9-25), so a linear scan is fine. Last entry holds
    // total_boot_segments, so the loop is well-bounded.
    int owning_tree = 0;
    for (int t = 0; t < num_trees; ++t) {
      if (tid >= boot_segment_offsets[t] && tid < boot_segment_offsets[t + 1]) {
        owning_tree = t;
        break;
      }
    }
    sub_trees[owning_tree]->insert(tid);
  }
}


//sanity check: are the VEB trees empty?
static __global__ void assert_empty(veb_tree ** segment_trees, int num_trees){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;


  for (int i =0; i< num_trees; i++){

    uint64_t alloc = segment_trees[i]->malloc_first();

    if (alloc != veb_tree::fail()){
      printf("Failed to clean VEB tree %d: Index %llu live\n", i, alloc);
    }
  }


}

// Wrap the boot kernels in an anonymous namespace so each translation unit
// gets its own internal-linkage instantiation. Without this, compute-sanitizer
// flags multi-TU consumers (e.g. Andes test binaries) with "Duplicate kernel
// entry: boot_shared_block_container<...>" — the linker is happy (weak symbols)
// but the sanitizer output becomes unusable noise.
namespace {

// Boot the per-tree pinned-block wavefront. Each tree's slot count was
// chosen at init time (see generate_on_device_impl), so we just read it
// out of per_size_pinned_blocks::num_blocks rather than passing a
// geometric/clamped recipe. tid 0..(num_blocks-1) of each tree fills
// that tree's slots; the rest of the grid no-ops for that tree.
//
// `boot_segment_offsets` is the same prefix-sum table consumed by
// boot_segments_deterministic. Each thread now knows exactly which
// segment it owns (deterministic), so it skips the random-walk through
// segment_tree and calls boot_block_deterministic with the pre-assigned
// segment_id.
template <typename allocator>
__global__ void boot_shared_block_container(
    allocator *alloc,
    uint16_t max_tree_id,
    const uint64_t *boot_segment_offsets) {
  uint64_t tid = gallatin::utils::get_tid();

  for (uint16_t tree_id = 0; tree_id < max_tree_id; tree_id++) {
    uint64_t slot_count =
        alloc->local_blocks->get_tree_local_blocks(tree_id)->num_blocks;
    if (tid < slot_count) {
      uint64_t segment_id = boot_segment_offsets[tree_id] + tid;
      alloc->boot_block_deterministic(tree_id, (int)tid, segment_id);
    }
  }
}



template <typename allocator>
__global__ void boot_shared_block_container_one_thread(allocator * alloc, uint16_t max_tree_id, int max_smid, int cutoff){

  uint64_t tid = gallatin::utils::get_tid();

  uint16_t tree_id = 0;

  if (tid != 0) return;


  while (tree_id < max_tree_id){

    for (int i = 0; i < max_smid; i++){

      alloc->boot_block(tree_id, i);

    }

    max_smid = max_smid/2;

    if (max_smid < cutoff) max_smid = cutoff;

    tree_id +=1;

  }

}

} // namespace

template <typename allocator>
__global__ void print_overhead_kernel(allocator * alloc){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  uint64_t overhead = alloc->calculate_overhead();

  printf("Allocator is using %llu bytes of overhead\n", overhead);

  return;

}

template <typename allocator>
__global__ void print_guided_fill_kernel(allocator * table, uint16_t id){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  table->print_guided_fill(id);

}

template <typename allocator>
__global__ void print_segment_fill_kernel(allocator * alloc){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= alloc->table->num_segments) return;

  //each tid scans for segment that is set

  uint16_t my_tree_id = alloc->table->read_tree_id(tid);

  if (my_tree_id == 65535) return;

  uint64_t num_blocks = alloc->table->get_blocks_per_segment(my_tree_id);

  uint64_t offset = tid*alloc->table->blocks_per_segment;

  uint64_t expected = 0;
  uint64_t free = 0;

  for (uint64_t i = 0; i < num_blocks; i++){

    auto my_block = alloc->table->get_block_from_global_block_id(offset+i);


    free += my_block->free_counter;
    expected+=4096;


  }

  if (free != expected){
    printf("Segment %lu has %lu/%lu allocations\n", tid, free, expected);
  }

}


// main allocator structure
// template arguments are
//  - size of each segment in bytes
//  - size of smallest segment allocatable
//  - size of largest segment allocatable
#ifdef GALLATIN_FLAT_BUFFER
// Phase B: per-tree pinned-slot descriptors cached in __constant__ memory, set
// once at init (gallatin_flat_publish). Collapses the per-malloc navigation
// chain (global_gallatin -> local_blocks -> block_containers[tree] -> blocks)
// into a single constant-cache read of {blocks base, num_blocks} for the tree.
// Single-instance (one live allocator per __constant__). Pairs with
// GALLATIN_CONST_BASE for constant-based address translation.
namespace gallatin_flat {
struct tree_slot_desc {
  Block **blocks;
  uint64_t num_blocks;
};
static constexpr int MAX_TREES = 16;
__constant__ tree_slot_desc g_tree_slots[MAX_TREES];
__device__ tree_slot_desc g_tree_slots_staging[MAX_TREES];
}  // namespace gallatin_flat

template <typename galloc_t>
__global__ void gallatin_flat_fill_kernel(galloc_t *alloc) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int nt = alloc->num_trees;
    if (nt > gallatin_flat::MAX_TREES) nt = gallatin_flat::MAX_TREES;
    for (int t = 0; t < nt; t++) {
      auto *st = alloc->local_blocks->get_tree_local_blocks(t);
      gallatin_flat::g_tree_slots_staging[t].blocks = st->blocks;
      gallatin_flat::g_tree_slots_staging[t].num_blocks = st->num_blocks;
    }
  }
}

template <typename galloc_t>
__host__ inline void gallatin_flat_publish(galloc_t *dev_alloc) {
  gallatin_flat_fill_kernel<<<1, 1>>>(dev_alloc);
  cudaDeviceSynchronize();
  gallatin_flat::tree_slot_desc tmp[gallatin_flat::MAX_TREES];
  cudaMemcpyFromSymbol(tmp, gallatin_flat::g_tree_slots_staging, sizeof(tmp));
  cudaMemcpyToSymbol(gallatin_flat::g_tree_slots, tmp, sizeof(tmp));
}
#endif

#ifdef GALLATIN_PERWARP_ATOMIC_DIAG
namespace gallatin_perwarp {
// per-warp, cacheline-padded scratch counters (diagnostic only): up to 64K
// warps, one counter per 128B line so distinct warps never share a sector.
__device__ unsigned int g_scratch[65536 * 32];
}  // namespace gallatin_perwarp
#endif

#ifdef GALLATIN_STATIC_COUNTER
// Phase D (increment 1): per-slot malloc counters live in a static device array
// (NOT inside the Block), so the reserving atomicAdd targets a static address
// with no Block* load. g_counter packs [generation:12 | count:20]; g_base/g_block
// shadow the underlying pinned block. Populated once at init (gallatin_static_publish).
// Generation is bumped on swap as an ABA/version guard. Single live allocator.
namespace gallatin_static {
static constexpr int MAX_TREES = 16;
#ifndef GALLATIN_MAX_N
#define GALLATIN_MAX_N 512
#endif
static constexpr int MAX_N = GALLATIN_MAX_N;  // max pinned slots per tree (must cover num_blocks)
#ifdef GALLATIN_PAD_COUNTERS
static constexpr int CSTRIDE = 32;  // 128B / sizeof(uint): one counter per cacheline
#else
static constexpr int CSTRIDE = 1;
#endif
__device__ unsigned int g_counter[MAX_TREES * MAX_N * 32];
__device__ uint64_t g_base[MAX_TREES * MAX_N];
__device__ void *g_block[MAX_TREES * MAX_N];
__device__ int g_nblk[MAX_TREES];
#ifdef GALLATIN_SLOT_BITMAP
// Per-slot reservation bitmap: 128 words (4096 bits = 4096 slices) per pinned
// slot. Lives in the slot storage, so overhead is constant (trees * slots *
// 128 words), independent of pool size / allocation size. The reserving atomicOr
// distributes across the 128 words -> an SM's warps stop funneling to one counter.
static constexpr int BM_WORDS = 128;
__device__ unsigned int g_bitmap[MAX_TREES * MAX_N * BM_WORDS];
#endif
#ifdef GALLATIN_SHARD_COUNTER
// Sharded monotonic counters: NSHARD counters per slot, each owning a
// contiguous SLICES_PER_SHARD range of the block's 4096 slices. O(1) reserve
// (atomicAdd, no scan, no extra load) AND distributed (warp hashes to a shard).
// Shard-major layout (k * trees*slots + slot) spreads a slot's shards across
// the array -> different L2 slices, no false sharing, no explicit padding.
static constexpr int NSHARD = 32;
static constexpr int SLICES_PER_SHARD = 4096 / NSHARD;  // 128
__device__ unsigned int g_shard[NSHARD * MAX_TREES * MAX_N];
#endif
}  // namespace gallatin_static

template <typename galloc_t>
__global__ void gallatin_static_fill_kernel(galloc_t *alloc) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nt = alloc->num_trees;
  if (nt > gallatin_static::MAX_TREES) nt = gallatin_static::MAX_TREES;
  for (int t = 0; t < nt; t++) {
    auto *st = alloc->local_blocks->get_tree_local_blocks(t);
    int n = (int)st->num_blocks;
    if (n > gallatin_static::MAX_N) n = gallatin_static::MAX_N;
    gallatin_static::g_nblk[t] = n;
    for (int s = 0; s < n; s++) {
      int idx = t * gallatin_static::MAX_N + s;
      Block *b = st->blocks[s];
      if (b != nullptr) {
        uint64_t gbid = alloc->table->get_global_block_offset(b);
        uint64_t base =
            (uint64_t)alloc->offset_to_allocation(gbid * 4096, (uint16_t)t);
        gallatin_static::g_block[idx] = b;
        gallatin_static::g_base[idx] = base;
        gallatin_static::g_counter[idx * gallatin_static::CSTRIDE] = 0u;  // gen 0, count 0
#ifdef GALLATIN_SLOT_BITMAP
        for (int w = 0; w < gallatin_static::BM_WORDS; w++)
          gallatin_static::g_bitmap[idx * gallatin_static::BM_WORDS + w] = 0u;
#endif
#ifdef GALLATIN_SHARD_COUNTER
        for (int k = 0; k < gallatin_static::NSHARD; k++)
          gallatin_static::g_shard[k * (gallatin_static::MAX_TREES *
                                        gallatin_static::MAX_N) + idx] = 0u;
#endif
      } else {
        gallatin_static::g_block[idx] = nullptr;
        gallatin_static::g_base[idx] = 0;
        gallatin_static::g_counter[idx * gallatin_static::CSTRIDE] = 4096u;  // empty slot -> forces refill
#ifdef GALLATIN_SLOT_BITMAP
        for (int w = 0; w < gallatin_static::BM_WORDS; w++)
          gallatin_static::g_bitmap[idx * gallatin_static::BM_WORDS + w] = 0xFFFFFFFFu;  // empty -> full -> fallback
#endif
#ifdef GALLATIN_SHARD_COUNTER
        for (int k = 0; k < gallatin_static::NSHARD; k++)
          gallatin_static::g_shard[k * (gallatin_static::MAX_TREES *
                                        gallatin_static::MAX_N) + idx] =
              gallatin_static::SLICES_PER_SHARD;  // empty -> full -> fallback
#endif
      }
    }
  }
}

template <typename galloc_t>
__host__ inline void gallatin_static_publish(galloc_t *dev_alloc) {
  gallatin_static_fill_kernel<<<1, 1>>>(dev_alloc);
  cudaDeviceSynchronize();
}
#endif

template <uint64_t bytes_per_segment, uint64_t smallest, uint64_t biggest>
struct Gallatin {
  using my_type = Gallatin<bytes_per_segment, smallest, biggest>;
  using sub_tree_type = veb_tree;
  using pinned_block_type = pinned_shared_blocks<smallest, biggest>;



  static_assert(bytes_per_segment >= biggest*4096);
  // internal structures
  veb_tree *segment_tree;

  alloc_table<bytes_per_segment, smallest> *table;

  sub_tree_type **sub_trees;

  pinned_block_type *local_blocks;

  int num_trees;

  int smallest_bits;

  // Per-tree segment-acquisition locks. Each lock lives in its own 128-byte
  // cache line so atomicOr/atomicAnd from threads in different trees do not
  // serialize through the same L2 line. Allocated as a separate buffer so the
  // padding doesn't bloat the Gallatin object itself (it lives in const cache
  // on the consumer side).
  struct alignas(128) padded_lock {
    uint v;
  };
  padded_lock *tree_locks;




  // Shared implementation for the three public generate_on_device variants.
  // memory_control selects the backing-memory kind (device / host-mapped /
  // managed). The three public wrappers below preserve the historical API.
  static __host__ my_type *generate_on_device_impl(
      uint64_t max_bytes, uint64_t seed, bool print_info,
      Gallatin_memory_type memory_control) {

    if (memory_control != device_only) {
      GPUErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
    }

    uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

    uint64_t num_trees =
        get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;

    // Hard error if the allocator is below the absolute floor — we can't
    // honor even one segment per tree.
    if (max_chunks < num_trees) {
      fprintf(stderr,
              "gallatin: allocator size %llu B = %llu segments is below the "
              "minimum of %llu (one segment per tree). Increase the size or "
              "shrink the (smallest, biggest) span.\n",
              (unsigned long long)max_bytes, (unsigned long long)max_chunks,
              (unsigned long long)num_trees);
      return nullptr;
    }

    // Pre-flight OOM check. The boot path issues a number of cudaMallocs
    // (segment memory, blocks, queues, chunk_ids, vEB layers, locks,
    // wavefront). On a near-full GPU these fail mid-boot, leaving the
    // allocator in a half-initialized state and producing confusing
    // downstream errors. Estimate the total request and fail fast with
    // a clear message if it can't fit.
    if (memory_control == device_only) {
      constexpr uint64_t blocks_per_segment =
          bytes_per_segment / (smallest * 4096);

      const uint64_t segment_memory_bytes = bytes_per_segment * max_chunks;
      const uint64_t blocks_bytes =
          sizeof(Block) * blocks_per_segment * max_chunks;
      const uint64_t queues_bytes =
          sizeof(Block *) * blocks_per_segment * max_chunks;
      const uint64_t chunk_ids_bytes = sizeof(uint16_t) * max_chunks;
      // vEB tree (segment_tree + num_trees sub_trees), tree_locks, and
      // wavefront storage are small in absolute terms but scale with
      // max_chunks. Conservative estimate: ~16 B per segment per tree
      // for vEB bitmaps + a few KB for the wavefront.
      const uint64_t veb_bytes =
          (uint64_t)(num_trees + 1) * (max_chunks / 4 + 64);
      const uint64_t misc_bytes = 4ULL * 1024 * 1024;  // round overhead pad

      const uint64_t required_bytes = segment_memory_bytes + blocks_bytes +
                                      queues_bytes + chunk_ids_bytes +
                                      veb_bytes + misc_bytes;

      size_t free_b = 0, total_b = 0;
      cudaMemGetInfo(&free_b, &total_b);

      if (required_bytes > free_b) {
        fprintf(stderr,
                "gallatin: OOM at boot — requested %.2f GB pool needs ~%.2f GB "
                "total GPU memory (segments + blocks + queues + bookkeeping), "
                "but only %.2f GB of %.2f GB is free on the device. Reduce "
                "the pool size by at least %.2f GB, or use "
                "Gallatin_memory_type::managed to spill to host.\n",
                (double)max_bytes / (1024.0 * 1024 * 1024),
                (double)required_bytes / (1024.0 * 1024 * 1024),
                (double)free_b / (1024.0 * 1024 * 1024),
                (double)total_b / (1024.0 * 1024 * 1024),
                (double)(required_bytes - free_b) / (1024.0 * 1024 * 1024));
        return nullptr;
      }
    }

    // Per-tree wavefront sizing. Start from the historical geometric
    // recipe (blocks_per_pinned_block=128, halve per tree, floor at
    // MIN_PINNED_CUTOFF), then cap each tree at (num_segments *
    // blocks_per_segment[tree]) / 4 so the wavefront takes at most ~1/4
    // of any tree's potential blocks — the rest stays available for
    // user allocations. The cap matters for large-slice trees where
    // each slot consumes a whole segment.
    constexpr uint64_t WAVEFRONT_BUDGET_FRACTION = 4;
    uint16_t *tree_slot_counts =
        gallatin::utils::get_host_version<uint16_t>(num_trees);

    uint64_t blocks_per_pinned_block = 128;
    uint64_t geom = blocks_per_pinned_block;
    bool any_reduced = false;
    // Deterministic boot claims one segment per slot, so the sum across
    // trees must fit in max_chunks. Track remaining pool capacity and
    // cap each tree's slot count against it. Without this, a small pool
    // (e.g. tests using 1 GB → 64 segments × 9 trees × MIN_PINNED_CUTOFF=32
    // wants 288) would exceed max_chunks and the boot would refuse to
    // start. Pre-deterministic-boot the random-walk path tolerated this
    // by simply leaving extra slots empty; deterministic boot needs the
    // budget tight up front.
    uint64_t remaining_pool = max_chunks;
    for (uint16_t t = 0; t < num_trees; ++t) {
      uint64_t blocks_per_seg =
          alloc_table<bytes_per_segment, smallest>::get_blocks_per_segment(t);
      uint64_t budget_cap =
          (max_chunks * blocks_per_seg) / WAVEFRONT_BUDGET_FRACTION;
      if (budget_cap == 0) budget_cap = 1;  // always at least one slot

      uint64_t target =
          geom < MIN_PINNED_CUTOFF ? (uint64_t)MIN_PINNED_CUTOFF : geom;
      uint64_t actual = target < budget_cap ? target : budget_cap;
      // Clamp to remaining pool capacity so the deterministic boot can
      // claim each slot's segment without overrunning the segment_tree.
      // Reserve at least 1 segment per remaining tree if possible, so
      // every tree gets at least one bootstrapped slot; otherwise leave
      // a tree with zero boot slots (it'll claim segments at first use).
      uint64_t trees_left = num_trees - t;
      uint64_t reserve_for_others = trees_left > 1 ? (trees_left - 1) : 0;
      uint64_t cap_from_pool = remaining_pool > reserve_for_others
                                   ? (remaining_pool - reserve_for_others)
                                   : 0;
      if (actual > cap_from_pool) actual = cap_from_pool;
      if (actual < target) any_reduced = true;
      tree_slot_counts[t] = (uint16_t)actual;
      remaining_pool -= actual;

      if (print_info && actual < target) {
        fprintf(stderr,
                "gallatin: tree %u (slice %llu B) wavefront reduced %llu "
                "-> %llu slots (%llu segments × %llu blocks/seg / %llu, "
                "remaining pool %llu)\n",
                (unsigned)t,
                (unsigned long long)
                    alloc_table<bytes_per_segment, smallest>::get_tree_alloc_size(t),
                (unsigned long long)target, (unsigned long long)actual,
                (unsigned long long)max_chunks,
                (unsigned long long)blocks_per_seg,
                (unsigned long long)WAVEFRONT_BUDGET_FRACTION,
                (unsigned long long)remaining_pool);
      }

      geom = geom / 2;
    }
    (void)any_reduced;

    my_type *host_version = get_host_version<my_type>();

    host_version->segment_tree =
        veb_tree::generate_on_device_nowait(max_chunks, seed);

    host_version->local_blocks =
        pinned_block_type::generate_on_device_nowait_per_tree(
            tree_slot_counts, num_trees);

    host_version->smallest_bits = get_first_bit_bigger(smallest);
    host_version->num_trees = num_trees;

    sub_tree_type **ext_sub_trees =
        get_host_version<sub_tree_type *>(num_trees);
    for (uint i = 0; i < num_trees; i++) {
      ext_sub_trees[i] =
          sub_tree_type::generate_on_device_nowait(max_chunks, i + seed);
    }
    host_version->sub_trees =
        move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

    // Deterministic boot setup: compute prefix-sum of per-tree slot counts
    // so each boot thread can be assigned a fixed segment_id. boot_segments_
    // deterministic uses this to (a) clear sub_trees as before, (b) clear
    // the [0, total_boot_segments) range of segment_tree (those segments
    // are now owned by trees), and (c) insert each boot segment into its
    // owning sub_tree.
    uint64_t *boot_segment_offsets_host =
        gallatin::utils::get_host_version<uint64_t>(num_trees + 1);
    boot_segment_offsets_host[0] = 0;
    for (uint16_t t = 0; t < num_trees; ++t) {
      boot_segment_offsets_host[t + 1] =
          boot_segment_offsets_host[t] + (uint64_t)tree_slot_counts[t];
    }
    uint64_t total_boot_segments = boot_segment_offsets_host[num_trees];
    // Invariant: the per-tree slot-count loop above clamps each tree's
    // tree_slot_counts[t] to the remaining pool capacity, so the prefix
    // sum total_boot_segments is always <= max_chunks. Keep a defensive
    // assert in debug builds; releasing builds skip the check.
    assert(total_boot_segments <= max_chunks);
    uint64_t *boot_segment_offsets_dev =
        gallatin::utils::move_to_device<uint64_t>(
            boot_segment_offsets_host, num_trees + 1);

    boot_segments_deterministic<<<(max_chunks - 1) / 512 + 1, 512>>>(
        host_version->segment_tree, host_version->sub_trees, max_chunks,
        num_trees, total_boot_segments, boot_segment_offsets_dev);

    #if GALLATIN_DEBUG_PRINTS
    cudaDeviceSynchronize();
    assert_empty<<<1, 1>>>(host_version->sub_trees, num_trees);
    cudaDeviceSynchronize();
    #endif

    // Per-tree locks, one per 128B cache line. Plus one extra slot at
    // index `num_trees` used by malloc_segment_allocation for the
    // global multi-segment grouping lock.
    host_version->tree_locks =
        gallatin::utils::get_device_version<padded_lock>(num_trees + 1);
    cudaMemset(host_version->tree_locks, 0,
               sizeof(padded_lock) * (num_trees + 1));

    host_version->table =
        alloc_table<bytes_per_segment, smallest>::generate_on_device_nowait(
            max_bytes, memory_control);

    if (print_info) {
      const char *kind = (memory_control == device_only)
                             ? "memory"
                             : (memory_control == host_only)
                                   ? "\033[1;32mpinned Host\033[1;0m memory"
                                   : "managed memory";
      printf("Booted Gallatin with %lu trees in range %lu-%lu and %f GB of %s "
             "%lu segments\n",
             num_trees, smallest, biggest,
             1.0 * max_bytes / 1024 / 1024 / 1024, kind, max_chunks);
    }

    auto device_version = move_to_device_nowait(host_version);

    // Launch enough threads to fill the largest tree's wavefront. Smaller
    // trees no-op past their own slot count via the per-tree num_blocks
    // check inside boot_shared_block_container.
    uint64_t max_slots = 0;
    for (uint16_t t = 0; t < num_trees; ++t) {
      if (tree_slot_counts[t] > max_slots) max_slots = tree_slot_counts[t];
    }
    cudaFreeHost(tree_slot_counts);

    constexpr int BOOT_BLOCK_DIM = 128;
    int boot_grid = (int)((max_slots + BOOT_BLOCK_DIM - 1) / BOOT_BLOCK_DIM);
    if (boot_grid < 1) boot_grid = 1;
    boot_shared_block_container<my_type>
        <<<boot_grid, BOOT_BLOCK_DIM>>>(device_version, (uint16_t)num_trees,
                                       boot_segment_offsets_dev);

    GPUErrorCheck(cudaDeviceSynchronize());

    // Cleanup: boot_segment_offsets_dev was a temporary; the boot kernels
    // have consumed it. (boot_segment_offsets_host was already freed by
    // move_to_device.)
    cudaFree(boot_segment_offsets_dev);

    return device_version;
  }

  // Device-backed allocator (the common case).
  static __host__ my_type *generate_on_device(uint64_t max_bytes, uint64_t seed,
                                              bool print_info = true) {
    return generate_on_device_impl(max_bytes, seed, print_info, device_only);
  }

  // Pinned-host-memory-backed allocator (mapped into the device address space).
  static __host__ my_type *generate_on_device_host(uint64_t max_bytes,
                                                   uint64_t seed,
                                                   bool print_info = true) {
    return generate_on_device_impl(max_bytes, seed, print_info, host_only);
  }

  // UVM/managed-memory-backed allocator.
  static __host__ my_type *generate_on_device_managed(uint64_t max_bytes,
                                                      uint64_t seed,
                                                      bool print_info = true) {
    return generate_on_device_impl(max_bytes, seed, print_info, managed);
  }

  // --- Legacy 4-arg overloads -------------------------------------------------
  // Calloc was removed (it was a failed dynamic-parallelism experiment); the
  // running_calloc flag is now ignored. These overloads exist to keep
  // downstream code that still passes the flag (e.g. andes' init wrapper)
  // compiling, but emit a compile-time deprecation warning so callers migrate.
  [[deprecated(
      "running_calloc was removed; calloc-mode no longer exists. Use the "
      "3-arg generate_on_device(bytes, seed, print_info).")]] static __host__
      my_type *
      generate_on_device(uint64_t max_bytes, uint64_t seed, bool print_info,
                         bool /*running_calloc*/) {
    return generate_on_device(max_bytes, seed, print_info);
  }

  [[deprecated(
      "running_calloc was removed; calloc-mode no longer exists. Use the "
      "3-arg generate_on_device_host(bytes, seed, print_info).")]] static __host__
      my_type *
      generate_on_device_host(uint64_t max_bytes, uint64_t seed,
                              bool print_info, bool /*running_calloc*/) {
    return generate_on_device_host(max_bytes, seed, print_info);
  }

  [[deprecated(
      "running_calloc was removed; calloc-mode no longer exists. Use the "
      "3-arg generate_on_device_managed(bytes, seed, "
      "print_info).")]] static __host__ my_type *
  generate_on_device_managed(uint64_t max_bytes, uint64_t seed,
                             bool print_info, bool /*running_calloc*/) {
    return generate_on_device_managed(max_bytes, seed, print_info);
  }

  // return the index of the largest bit set
  static __host__ __device__ int get_first_bit_bigger(uint64_t counter) {
    return gallatin::utils::get_first_bit_bigger(counter);
  }

  // get number of sub trees live
  static __host__ __device__ int get_num_trees() {
    return get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;
  }

  // return memory used to device
  static __host__ void free_on_device(my_type *dev_version) {
    // this frees dev version.
    my_type *host_version = move_to_host<my_type>(dev_version);

    uint64_t num_trees =
        get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;

    sub_tree_type **host_subtrees =
        move_to_host<sub_tree_type *>(host_version->sub_trees, num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      sub_tree_type::free_on_device(host_subtrees[i]);
    }

    alloc_table<bytes_per_segment, smallest>::free_on_device(
        host_version->table);


    veb_tree::free_on_device(host_version->segment_tree);

    pinned_block_type::free_on_device(host_version->local_blocks);

    cudaFree(host_version->tree_locks);

    cudaFreeHost(host_subtrees);

    cudaFreeHost(host_version);
  }


  //simple spot check to make sure that the allocator isn't giving off memory
  //outside of it's addresses.
  __device__ bool check_alloc_valid(void * allocation){

    uint64_t byte_difference = (uint64_t) allocation - (uint64_t) table->memory;

    uint64_t max_bytes = (table->num_segments) * bytes_per_segment;

    printf("%llx inside of %llx\n", (uint64_t) allocation, (uint64_t) table->memory);

    if (byte_difference > max_bytes){

      printf("Byte difference: %lu > %lu\n", byte_difference, max_bytes);
      return false;
    }

    printf("Byte difference: %lu <= %lu\n", byte_difference, max_bytes);
    return true;

  }

  __device__ uint64_t get_capacity_bytes(){
    return table->num_segments * bytes_per_segment;
  }

  // given a pointer, return the segment it belongs to
  // __device__ inline uint64_t snap_pointer_to_block(void *ext_ptr) {


  //   char *memory_start = table->get_segment_memory_start(segment);

  //   uint64_t snapped_offset =
  //       ((uint64_t)(ext_ptr - memory_start)) / bytes_per_segment;

  //   return snapped_offset;
  // }

  // Cast an offset back into a memory pointer
  // this requires the offset and the tree_id so that we know how far to scale
  // the pointer
  __device__ void *alloc_offset_to_ptr(uint64_t offset, uint16_t tree_id) {
    uint64_t block_id = offset / 4096;

    uint64_t relative_offset = offset % 4096;

    uint64_t segment = block_id / table->blocks_per_segment;

    uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    char *memory_start = table->get_segment_memory_start(segment);

    // with start of segment and alloc size, we can set the pointer relative to
    // the segment
    return (void *)(memory_start + relative_offset * alloc_size);
  }



  // Initialize one pinned-wavefront slot at boot. The wavefront is a perf
  // optimization, not a hard requirement: if the allocator was sized too
  // small to pre-populate every slot for every tree (most commonly with
  // large-slice trees, where one slot consumes one segment), we simply
  // leave that slot empty. The malloc path's get_valid_block already
  // walks forward when it sees a null slot, so functionally the allocator
  // just runs with a sparser wavefront for that tree.
  // Deterministic boot path. The caller (boot_shared_block_container) has
  // already computed `segment_id` from the prefix-sum table and the kernel
  // boot_segments_deterministic has already cleared this segment from
  // segment_tree and inserted it into sub_trees[tree_id]. We just need to
  // initialize the segment's metadata, pull a block from it, and publish
  // into the per-tree pinned slot.
  //
  // No segment_tree contention, no random walk, no retry loop — boot is
  // O(threads). This is the sanitizer-friendly path (the random-walk
  // path livelocks under compute-sanitizer instrumentation overhead on
  // Blackwell when MIN_PINNED_CUTOFF=32 inflates the boot grid 8x).
  __device__ void boot_block_deterministic(uint16_t tree_id, int smid,
                                           uint64_t segment_id) {
    per_size_pinned_blocks *local_shared_block_storage =
        local_blocks->get_tree_local_blocks(tree_id);

    if (smid >= (int)local_shared_block_storage->num_blocks) {
      // Bug in the boot kernel — tid range exceeds slot count.
      printf("ERR boot_block_deterministic %d >= %llu\n", smid,
             local_shared_block_storage->num_blocks);
      #if GALLATIN_TRAP_ON_ERR
      asm volatile("trap;");
      #endif
      return;
    }

    // Each thread has a unique segment_id → no contention on chunk_ids
    // (other than the natural per-32-bit-word atomic; uint16_t still works).
    if (!table->setup_segment(segment_id, tree_id)) {
      // setup_segment's release-CAS only fails if the slot wasn't ~0 at
      // entry. Pre-boot cudaMemset(~0U) guarantees it is, so this is a
      // hard invariant violation if it triggers.
      printf("Boot (deterministic): setup_segment failed for segment %llu (tree %u)\n",
             (unsigned long long)segment_id, (unsigned)tree_id);
      #if GALLATIN_TRAP_ON_ERR
      asm volatile("trap;");
      #endif
      return;
    }

    bool last_block = false;
    Block *new_block = table->get_block(segment_id, tree_id, last_block);

    if (new_block == nullptr) {
      // Should not happen for a freshly-claimed segment.
      printf("Boot (deterministic): get_block returned nullptr for "
             "segment %llu (tree %u)\n",
             (unsigned long long)segment_id, (unsigned)tree_id);
      #if GALLATIN_TRAP_ON_ERR
      asm volatile("trap;");
      #endif
      return;
    }

    // If this boot grab consumed the segment's only block — true for the
    // largest tree, where blocks_per_segment == 1 — the segment is now
    // exhausted. boot_segments_deterministic published it into sub_trees[tree],
    // so we must detach it here, mirroring the runtime last_block path in
    // request_new_block_from_tree. Otherwise the segment stays listed with no
    // free block and the first runtime request livelocks re-selecting it.
    if (last_block) {
      sub_trees[tree_id]->remove(segment_id);
      __threadfence();
    }

    if (!local_shared_block_storage->swap_out_nullptr(smid, new_block)) {
      printf("Boot (deterministic): slot %d for tree %u already initialized\n",
             smid, (unsigned)tree_id);
      #if GALLATIN_TRAP_ON_ERR
      asm volatile("trap;");
      #endif
    }
  }

  __device__ void boot_block(uint16_t tree_id, int smid){

    per_size_pinned_blocks * local_shared_block_storage =
          local_blocks->get_tree_local_blocks(tree_id);


    if (smid >= local_shared_block_storage->num_blocks){
      // Bug in the boot kernel — tid range exceeds slot count. Real error.
      printf("ERR %d >= %llu\n", smid, local_shared_block_storage->num_blocks);

      #if GALLATIN_TRAP_ON_ERR
      asm volatile ("trap;");
      #endif

      return;

    }

  	Block * new_block = request_new_block_from_tree(tree_id);

  	if (new_block == nullptr){
        // No segment available for this tree at boot. Leave the slot empty
        // (it stays at its post-cudaMemset nullptr state); subsequent
        // mallocs that target this slot will probe forward in
        // get_valid_block, find a populated peer, and proceed. No need
        // to trap — this happens deterministically on small-allocator
        // configurations and is recoverable.
        return;
  	}


    uint64_t alt_block_segment = table->get_segment_from_block_ptr(new_block);

    uint16_t alt_tree_id = table->read_tree_id(alt_block_segment);


    uint64_t block_id = table->get_global_block_offset(new_block);

    if (tree_id != alt_tree_id){
        //this triggers, yeet
        printf("Boot Block %llu: segment %llu not init for malloc: %u != %u\n", block_id, alt_block_segment, tree_id, alt_tree_id);

        #if GALLATIN_TRAP_ON_ERR
        asm volatile ("trap;");
        #endif

      }



    if(!local_shared_block_storage->swap_out_nullptr(smid, new_block)){

    	printf("Error: Block in position %d for tree %d already initialized\n", smid, tree_id);

      #if GALLATIN_TRAP_ON_ERR
      asm volatile ("trap;");
      #endif

    }

  }


  __device__ uint64_t malloc_segment_allocation(uint & num_segments_required){

    // Fast bail: no free segments available
    if (segment_tree->is_empty()) return ~0ULL;

    // Warp-level coalesce: threads in this warp that all want the same
    // number of segments do ONE gather for (team_size * K) segments,
    // then each thread takes its own K-segment slice and publishes its
    // own tree_id. This collapses N lock acquisitions to 1 and gives the
    // vEB tree one packed scan instead of N fragmented ones — both wins
    // when multiple warp lanes simultaneously request big allocations.
    cg::coalesced_group warp_team = cg::coalesced_threads();
    cg::coalesced_group team =
        labeled_partition(warp_team, (uint)num_segments_required);
    const uint team_size = team.size();
    const uint my_rank = team.thread_rank();
    const uint64_t total_segments =
        (uint64_t)team_size * (uint64_t)num_segments_required;

    // Lock slot at index `num_trees` is the global multi-segment grouping
    // lock — separate from per-tree locks. tree_locks is allocated with
    // num_trees+1 entries for exactly this reason.
    uint64_t group_alloc_index = veb_tree::fail();
    if (my_rank == 0) {
      while (!acquire_tree_lock(num_trees));
      group_alloc_index = segment_tree->gather_multiple(total_segments);
      release_tree_lock(num_trees);
    }
    group_alloc_index = team.shfl(group_alloc_index, 0);

    if (group_alloc_index == veb_tree::fail()) {
      return ~0ULL;
    }

    // Each thread carves out its own contiguous slice from the group.
    const uint64_t alloc_index =
        group_alloc_index + (uint64_t)my_rank * num_segments_required;

    // Each thread publishes its own first-segment tree_id (no inter-thread
    // serialization — each thread's first segment is distinct).
    if (!table->set_tree_id(alloc_index,
                            num_trees + 1 + num_segments_required)) {
      // set_tree_id is a release-CAS from ~0; failure means this thread's
      // first-segment slot wasn't ~0 when we tried to publish. In correct
      // operation impossible (we just claimed it via gather_multiple).
      // Return *our* slice to segment_tree; sibling threads keep theirs.
      segment_tree->return_multiple(alloc_index, num_segments_required);

      #if GALLATIN_DEBUG_PRINTS
      printf("malloc_segment_allocation: set_tree_id raced on segment %llu "
             "(span=%u). Returned segments to tree; alloc failed.\n",
             (unsigned long long)alloc_index, num_segments_required);
      #endif
      #if GALLATIN_TRAP_ON_ERR
      asm volatile ("trap;");
      #endif

      return ~0ULL;
    }

    return alloc_index * table->blocks_per_segment * 4096;
  }
  
  __device__ uint64_t malloc_block_allocation(uint16_t & tree_id){

    // #if GALLATIN_DEBUG_PRINTS
    // printf("Alloc of %llu bytes pulling from block in tree %d\n", bytes_needed, block_tree);
    // #endif

    Block * my_block = request_new_block_from_tree(tree_id);

    if (my_block == nullptr){
      return ~0ULL;
    }

    uint64_t global_block_id = table->get_global_block_offset(my_block);

    uint old = my_block->malloc_fill_block(tree_id);

    if (old != 0){

      #if GALLATIN_DEBUG_PRINTS
      printf("Block was already set %u\n", old);
      #endif


      free_offset(global_block_id*4096);

      return ~0ULL;

    }


    return global_block_id*4096;


  }


  // Fast path: acquire a single slice from a tree.
  //
  // This is specialized for alloc_count == 1 (the overwhelming common case)
  // and skips three sources of overhead that the general malloc_slice_allocation
  // pays for the rare multi-slice case:
  //   1. cg::exclusive_scan over alloc_count (log2(team_size) shuffles).
  //   2. block_correct_frees reduce + conditional atomicAdd.
  //   3. The four-way validity classification (start_valid/end_valid math).
  //
  // For alloc_count == 1, the per-thread allocation index is simply
  // (true_count + thread_rank) and the "this batch crossed 4096" check
  // collapses to a single ballot on (allocation == 4095).
  __device__ uint64_t malloc_slice_one(uint16_t tree_id) {

    per_size_pinned_blocks *local_shared_block_storage =
        local_blocks->get_tree_local_blocks(tree_id);

    int shared_block_storage_index;
    Block *my_block;

    int num_attempts = 0;

    while (num_attempts < GALLATIN_MAX_ATTEMPTS) {

      cg::coalesced_group full_warp_team = cg::coalesced_threads();
      cg::coalesced_group coalesced_team =
          labeled_partition(full_warp_team, tree_id);

      if (coalesced_team.thread_rank() == 0) {
        my_block = local_shared_block_storage->get_valid_block(
            shared_block_storage_index);
      }
      shared_block_storage_index =
          coalesced_team.shfl(shared_block_storage_index, 0);
      my_block = coalesced_team.shfl(my_block, 0);

      if (my_block == nullptr) {
        if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty()) {
          return ~0ULL;
        }
        num_attempts++;
        continue;
      }

      uint64_t global_block_id = table->get_global_block_offset(my_block);

      // Single atomicAdd by thread 0, broadcast via shfl. Result has the
      // tree-id tag in bits 20..31 and the running malloc count in bits 0..19.
      uint merged_count = my_block->block_malloc_tree(coalesced_team);
      uint true_count = merged_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);

      uint thread_rank = coalesced_team.thread_rank();
      uint64_t allocation = true_count + thread_rank;
      bool valid = allocation < 4096;

      // The thread that lands exactly on the last in-bounds slot (4095)
      // is the sole replacer: per coalesced atomicAdd batch, at most one
      // thread's allocation is == 4095. No ballot needed — the thread
      // either is or isn't that thread.
      if (allocation == 4095) {
        replace_block(tree_id, shared_block_storage_index, my_block,
                      local_shared_block_storage);
      }
      coalesced_team.sync();

      if (valid) {
        if (!my_block->check_valid(merged_count, tree_id)) {
          // Tree-id tag mismatch: someone reformatted this block. Roll back.
          free_offset(allocation + global_block_id * 4096);
        } else {
          return allocation + global_block_id * 4096;
        }
      }

      num_attempts++;
    }

    return ~0ULL;
  }

  // Single-thread fast path: same slice reservation as malloc_slice_one but for
  // a caller that allocates one object from one lane (e.g. a data structure
  // allocating one node per warp-op). Skips cg::coalesced_threads() +
  // labeled_partition + the shfl broadcasts, which are pure overhead when the
  // active group is a single thread. MUST be called by a single thread.
  __device__ uint64_t malloc_slice_one_single(uint16_t tree_id) {

    per_size_pinned_blocks *local_shared_block_storage =
        local_blocks->get_tree_local_blocks(tree_id);

    int shared_block_storage_index;
    int num_attempts = 0;

    while (num_attempts < GALLATIN_MAX_ATTEMPTS) {

      Block *my_block = local_shared_block_storage->get_valid_block(
          shared_block_storage_index);

      if (my_block == nullptr) {
        if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty()) {
          return ~0ULL;
        }
        num_attempts++;
        continue;
      }

      uint64_t global_block_id = table->get_global_block_offset(my_block);

      uint merged_count =
          atomicAdd((unsigned int *)&my_block->malloc_counter, 1u);
      uint true_count = merged_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);

      uint64_t allocation = true_count;  // single thread, rank 0
      bool valid = allocation < 4096;

      if (allocation == 4095) {
        replace_block(tree_id, shared_block_storage_index, my_block,
                      local_shared_block_storage);
      }

      if (valid) {
        if (!my_block->check_valid(merged_count, tree_id)) {
          free_offset(allocation + global_block_id * 4096);
        } else {
          return allocation + global_block_id * 4096;
        }
      }

      num_attempts++;
    }

    return ~0ULL;
  }

  // Single-thread malloc. Routes the common single-slice case through the
  // uncoalesced fast path; falls back to the cooperative malloc() for larger
  // sizes and on transient failure. MUST be called by a single thread.
#ifdef GALLATIN_STATIC_COUNTER
  // Phase D: retire a full static slot — stamp the old block full so its
  // free/recycle accounting still matches, pull a fresh block into the
  // underlying pinned slot, then republish base + bump generation (release).
  __device__ void swap_static_slot(uint16_t tree_id, int idx, int slot) {
    Block *old = reinterpret_cast<Block *>(gallatin_static::g_block[idx]);
    per_size_pinned_blocks *container = local_blocks->get_tree_local_blocks(tree_id);
    if (old != nullptr) {
      atomicExch((unsigned int *)&old->malloc_counter,
                 (((unsigned int)tree_id) << GALLATIN_BLOCK_TREE_OFFSET) | 4096u);
      replace_block(tree_id, slot, old, container);
    }
    Block *fresh = gallatin::utils::load_acquire(&container->blocks[slot]);
    unsigned int next_gen =
        (gallatin_static::g_counter[idx * gallatin_static::CSTRIDE] >> GALLATIN_BLOCK_TREE_OFFSET) + 1u;
    if (fresh != nullptr) {
      uint64_t gbid = table->get_global_block_offset(fresh);
      gallatin_static::g_block[idx] = fresh;
      gallatin_static::g_base[idx] =
          (uint64_t)offset_to_allocation(gbid * 4096, tree_id);
      __threadfence();  // publish base before the generation bump
      atomicExch(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE],
                 next_gen << GALLATIN_BLOCK_TREE_OFFSET);  // count 0, new gen
    } else {
      gallatin_static::g_block[idx] = nullptr;
      atomicExch(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE],
                 (next_gen << GALLATIN_BLOCK_TREE_OFFSET) | 4096u);
    }
  }

  // Phase D hot path: atomicAdd on a STATIC per-slot counter (no Block* load).
  // The acquire-load of g_counter pairs with swap's release-exch; gen==gen0
  // confirms no swap occurred in the window, so `base` matches the slice.
  // Single-thread (call from one lane). Assumes pow2 nblk (true at boot).
  __device__ void *malloc_static(uint64_t size) {
    uint16_t tree_id = get_tree_id_from_size(size);
    if (tree_id >= num_trees) return malloc(size);
    const uint64_t alloc_size = table->get_tree_alloc_size(tree_id);
    int nblk = gallatin_static::g_nblk[tree_id];
    int base_idx = tree_id * gallatin_static::MAX_N;

    int attempts = 0;
    while (attempts < GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS) {
#ifdef GALLATIN_SPREAD_ATOMIC
      int slot = (gallatin::utils::get_smid() + (int)((threadIdx.x >> 5) * 17u)) &
                 (nblk - 1);
#else
      int slot = gallatin::utils::get_smid() & (nblk - 1);
#endif
      int idx = base_idx + slot;

      unsigned int gen0 =
          gallatin::utils::load_acquire(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE]) >>
          GALLATIN_BLOCK_TREE_OFFSET;
      uint64_t base = gallatin_static::g_base[idx];
      unsigned int merged = atomicAdd(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE], 1u);
      unsigned int count = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      unsigned int gen = merged >> GALLATIN_BLOCK_TREE_OFFSET;

      if (gen == gen0 && count < 4096) {
        if (count == 4095) {
          swap_static_slot(tree_id, idx, slot);
        }
        return reinterpret_cast<void *>(base + (uint64_t)count * alloc_size);
      }
      attempts++;
    }
    return malloc(size);
  }

  // Phase D increment 2: cached static counter. Caches {slot idx, base, gen} in
  // the caller's context. Steady state = ONE atomicAdd on the static counter at
  // a cached index, ZERO loads (no gen0 acquire-load). Refill (rare) re-resolves
  // the slot with the gen-checked path. Single-thread.
  // tree_id + alloc_size are hoisted in by the caller (loop-invariant for a
  // fixed-size allocator) so the hot path issues ZERO resolution loads.
  __device__ void *malloc_static_cached(int &cidx, uint64_t &cbase,
                                        unsigned int &cgen, uint16_t tree_id,
                                        uint64_t alloc_size, uint64_t size) {
    if (tree_id >= num_trees) return malloc(size);

    if (cidx >= 0) {
      unsigned int merged = atomicAdd(&gallatin_static::g_counter[cidx * gallatin_static::CSTRIDE], 1u);
      unsigned int count = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      unsigned int gen = merged >> GALLATIN_BLOCK_TREE_OFFSET;
      if (gen == cgen && count < 4096) {
        void *p = reinterpret_cast<void *>(cbase + (uint64_t)count * alloc_size);
        if (count == 4095) {
          swap_static_slot(tree_id, cidx,
                           cidx - tree_id * gallatin_static::MAX_N);
          cidx = -1;
        }
        return p;
      }
      cidx = -1;  // gen mismatch (swapped) or full -> refill
    }

    int nblk = gallatin_static::g_nblk[tree_id];
    int base_idx = tree_id * gallatin_static::MAX_N;
    int attempts = 0;
    while (attempts < GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS) {
#ifdef GALLATIN_SPREAD_ATOMIC
      int slot = (gallatin::utils::get_smid() + (int)((threadIdx.x >> 5) * 17u)) &
                 (nblk - 1);
#else
      int slot = gallatin::utils::get_smid() & (nblk - 1);
#endif
      int idx = base_idx + slot;
      unsigned int gen0 =
          gallatin::utils::load_acquire(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE]) >>
          GALLATIN_BLOCK_TREE_OFFSET;
      uint64_t base = gallatin_static::g_base[idx];
      unsigned int merged = atomicAdd(&gallatin_static::g_counter[idx * gallatin_static::CSTRIDE], 1u);
      unsigned int count = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      unsigned int gen = merged >> GALLATIN_BLOCK_TREE_OFFSET;
      if (gen == gen0 && count < 4096) {
        void *p = reinterpret_cast<void *>(base + (uint64_t)count * alloc_size);
        if (count == 4095) {
          swap_static_slot(tree_id, idx, slot);
        } else {
          cidx = idx;
          cbase = base;
          cgen = gen;  // cache for the next allocation
        }
        return p;
      }
      attempts++;
    }
    return malloc(size);
  }

#ifdef GALLATIN_MINIMAL
  // MINIMAL Gallatin-pool allocate: warp-private static counter over the
  // pre-pinned blocks, referencing NONE of the cold machinery (no malloc()
  // fallback, no replace_block, no refill loop) so nvcc has nothing big to
  // inline -> the kernel collapses toward slab's ~160 instructions -> ~19 regs
  // -> high occupancy. ALLOC_SIZE is a compile-time immediate (slice*size ->
  // shift). Caches {slot,base}. Returns nullptr if a slot block overflows
  // (benchmark pre-sizes MIN_PINNED so warp-private slots never fill).
  template <uint64_t ALLOC_SIZE>
  __device__ void *malloc_minimal(int &cidx, uint64_t &cbase, uint16_t tree_id) {
    if (cidx >= 0) {
      unsigned int count =
          atomicAdd(&gallatin_static::g_counter[cidx * gallatin_static::CSTRIDE],
                    1u) & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      if (count < 4096)
        return reinterpret_cast<void *>(cbase + (uint64_t)count * ALLOC_SIZE);
      cidx = -1;
    }
    unsigned int gwid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int nblk = gallatin_static::g_nblk[tree_id];
    int slot =
        tree_id * gallatin_static::MAX_N + (int)(gwid & (unsigned)(nblk - 1));
    unsigned int count =
        atomicAdd(&gallatin_static::g_counter[slot * gallatin_static::CSTRIDE],
                  1u) & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
    if (count < 4096) {
      cidx = slot;
      cbase = gallatin_static::g_base[slot];
      return reinterpret_cast<void *>(cbase + (uint64_t)count * ALLOC_SIZE);
    }
    return nullptr;
  }
#endif

#ifdef GALLATIN_SLOT_BITMAP
  // Per-slot bitmap reservation. The reserving atomicOr is distributed across
  // BM_WORDS words per slot, so an SM's warps spread instead of funneling to one
  // counter (the proven contention cost). Caches {slot idx, base, word} in the
  // caller's context; tree_id + alloc_size hoisted in (loop-invariant).
  // v1: no swap -- on block-full, fall back to cooperative malloc(). Single-thread.
  __device__ void *malloc_slot_bitmap(int &cidx, uint64_t &cbase, int &cword,
                                      uint16_t tree_id, uint64_t alloc_size,
                                      uint64_t size) {
    if (tree_id >= num_trees) return malloc(size);
    // global warp id: spread the starting word across ALL BM_WORDS (not just
    // the 0..7 warp-in-block id) so warps don't pile onto a few words.
    unsigned int gwid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (cidx < 0) {
      int nblk = gallatin_static::g_nblk[tree_id];
      int slot = (gallatin::utils::get_smid()
#ifdef GALLATIN_SPREAD_ATOMIC
                  + (int)((threadIdx.x >> 5) * 17u)
#endif
                  ) & (nblk - 1);
      cidx = tree_id * gallatin_static::MAX_N + slot;
      cbase = gallatin_static::g_base[cidx];
#ifdef GALLATIN_SHARD_COUNTER
      cword = (int)(gwid & (gallatin_static::NSHARD - 1));
#else
      cword = (int)(gwid & (gallatin_static::BM_WORDS - 1));
#endif
    }
#ifdef GALLATIN_SHARD_COUNTER
    // sharded monotonic counters: O(1) atomicAdd on the warp's cached shard,
    // distributed across NSHARD per-slot counters (shard-major -> no false share).
    const int TS = gallatin_static::MAX_TREES * gallatin_static::MAX_N;
    for (int probe = 0; probe < gallatin_static::NSHARD; probe++) {
      int k = cword & (gallatin_static::NSHARD - 1);
      unsigned int count =
          atomicAdd(&gallatin_static::g_shard[k * TS + cidx], 1u);
      if (count < (unsigned)gallatin_static::SLICES_PER_SHARD) {
        uint64_t slice =
            (uint64_t)k * gallatin_static::SLICES_PER_SHARD + count;
        return reinterpret_cast<void *>(cbase + slice * alloc_size);
      }
      cword = (cword + 1) & (gallatin_static::NSHARD - 1);  // shard full, advance
    }
    cidx = -1;  // block full -> refill next call
    return malloc(size);
#else
    // per-warp bit rotation: start the in-word search at a warp-dependent bit
    // so warps sharing a word don't all collide on __ffs's lowest free bit.
    const unsigned int brot = gwid & 31u;
    unsigned int *bm =
        &gallatin_static::g_bitmap[(uint64_t)cidx * gallatin_static::BM_WORDS];
    for (int probe = 0; probe < gallatin_static::BM_WORDS; probe++) {
      int w = cword & (gallatin_static::BM_WORDS - 1);
      unsigned int word = bm[w];
      while (word != 0xFFFFFFFFu) {
        // rotate free-bit selection by brot to de-correlate warps on this word.
        unsigned int free = ~word;
        unsigned int rotfree = (free >> brot) | (free << (32u - brot));
        int rbit = __ffs(rotfree) - 1;
        int bit = (int)((rbit + brot) & 31u);
        unsigned int mask = 1u << bit;
        unsigned int old = atomicOr(&bm[w], mask);
        if (!(old & mask)) {
          uint64_t slice = (uint64_t)w * 32u + (uint64_t)bit;
          return reinterpret_cast<void *>(cbase + slice * alloc_size);
        }
        word = old | mask;  // bit was taken; try another bit in this word
      }
      cword = (cword + 1) & (gallatin_static::BM_WORDS - 1);  // word full, advance
    }
    cidx = -1;  // block full -> refill next call
    return malloc(size);
#endif
  }
#endif
#endif

#ifdef GALLATIN_CACHED_BLOCK
#ifdef GALLATIN_NOINLINE_REFILL
#define GALLATIN_REFILL_ATTR __noinline__
#else
#define GALLATIN_REFILL_ATTR __forceinline__
#endif
#ifdef GALLATIN_OUTLINE_COLD
  // Outline the giant cooperative malloc() fallback so it does NOT get inlined
  // into alloc_kernel and inflate its register footprint / cause spilling. It
  // runs only under genuine pool pressure (essentially never on the hot path).
  __noinline__ __device__ void *malloc_cold(uint64_t size) { return malloc(size); }
#define GALLATIN_COLD_MALLOC(sz) malloc_cold(sz)
#else
#define GALLATIN_COLD_MALLOC(sz) malloc(sz)
#endif
#ifdef GALLATIN_OUTLINE_ALL
  // Outline the rare block-swap (hit ~1/4096 allocs) so replace_block + the
  // pinned-buffer machinery don't inline into the hot kernel (the SASS bloat
  // that pins registers/occupancy). __noinline__ keeps alloc_kernel tiny.
  __noinline__ __device__ void replace_block_cold(int tree_id, int csmid,
                                                  Block *cblk) {
    per_size_pinned_blocks *container =
        local_blocks->get_tree_local_blocks(tree_id);
    replace_block(tree_id, csmid, cblk, container);
  }
#endif
  // Cold refill for malloc_cached, optionally outlined (__noinline__) so its
  // 64-bit locals don't inflate the whole fast-path/kernel register count.
  GALLATIN_REFILL_ATTR __device__ void *malloc_cached_refill(
      void *&cblk_raw, char *&cbase, int &csmid, uint16_t tree_id,
      uint64_t alloc_size, uint64_t size) {
    // ---- refill: pull a block from the flat slot table, then cache it ----
    Block **slots = gallatin_flat::g_tree_slots[tree_id].blocks;
    uint64_t nblk = gallatin_flat::g_tree_slots[tree_id].num_blocks;
    int num_attempts = 0;
    while (num_attempts < GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS) {
#ifdef GALLATIN_WARP_PRIVATE_BLOCK
      // map each warp to a (near-)private block by global warp id, so warps
      // stop sharing one per-SM block counter (reaches the per-warp-private
      // contention-free floor when nblk >= concurrent warps).
      int base_smid = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
#elif defined(GALLATIN_SPREAD_ATOMIC)
      int base_smid =
          gallatin::utils::get_smid() + (int)((threadIdx.x >> 5) * 17u);
#else
      int base_smid = gallatin::utils::get_smid();
#endif
#ifdef GALLATIN_POW2_BLOCKS
      int smid = base_smid & (int)(nblk - 1);
#else
      int smid = base_smid % nblk;
#endif
      Block *my_block = gallatin::utils::load_acquire(&slots[smid]);
      int probe = 0;
      while (my_block == nullptr && probe < SHARED_BLOCK_COUNTER_CUTOFF) {
#ifdef GALLATIN_POW2_BLOCKS
        smid = (smid + 1) & (int)(nblk - 1);
#else
        smid = (smid + 1) % nblk;
#endif
        my_block = gallatin::utils::load_acquire(&slots[smid]);
        probe++;
      }
      if (my_block == nullptr) {
        if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty())
          return GALLATIN_COLD_MALLOC(size);
        num_attempts++;
        continue;
      }
      uint64_t gbid = table->get_global_block_offset(my_block);
      char *block_base =
          reinterpret_cast<char *>(offset_to_allocation(gbid * 4096, tree_id));
      uint merged = atomicAdd((unsigned int *)&my_block->malloc_counter, 1u);
      uint slice = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      if (slice == 4095) {
        per_size_pinned_blocks *container =
            local_blocks->get_tree_local_blocks(tree_id);
        replace_block(tree_id, smid, my_block, container);
      }
      if (slice < 4096) {
        if (!my_block->check_valid(merged, tree_id)) {
          free_offset(slice + gbid * 4096);
        } else {
          if (slice < 4095) {  // cache only a block that still has slices
            cblk_raw = my_block;
            cbase = block_base;
            csmid = smid;
          }
          return block_base + static_cast<uint64_t>(slice) * alloc_size;
        }
      }
      num_attempts++;
    }
    return GALLATIN_COLD_MALLOC(size);
  }

  // Phase C': register/context-cached active block. Steady state is ONE
  // atomicAdd on the cached block with NO preceding slot load (the slab model);
  // the ld.acquire(slot) happens only on refill. Stale-safe via tag-in-counter:
  // if the cached block was recycled to another size, the atomicAdd returns a
  // foreign tag -> roll back the phantom increment and refill. Requires
  // GALLATIN_FLAT_BUFFER (slot table) + GALLATIN_CONST_BASE. Single-thread.
  __device__ void *malloc_cached(void *&cblk_raw, char *&cbase, int &csmid,
                                 uint64_t size) {
    uint16_t tree_id = get_tree_id_from_size(size);
    if (tree_id >= num_trees) return malloc(size);
    const uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    // ---- fast path: reserve from the cached block, no slot load ----
    Block *cblk = reinterpret_cast<Block *>(cblk_raw);
    if (cblk != nullptr) {
#ifdef GALLATIN_PERWARP_ATOMIC_DIAG
      // DIAGNOSTIC ONLY (unsafe): point the reserving atomic at a per-warp,
      // cacheline-padded private counter -> ZERO cross-warp contention, every
      // other instruction/load identical to cp2. Isolates the cost due purely
      // to atomic contention on the shared block counter.
      unsigned int gwid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
      uint merged =
          atomicAdd(&gallatin_perwarp::g_scratch[(gwid & 65535u) * 32u], 1u);
      uint slice = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      bool ok = true;
#else
      uint merged = atomicAdd((unsigned int *)&cblk->malloc_counter, 1u);
      uint slice = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      bool ok = cblk->check_valid(merged, tree_id);
#endif
      if (ok && slice < 4096) {
        void *p = cbase + static_cast<uint64_t>(slice) * alloc_size;
        if (slice == 4095) {
          per_size_pinned_blocks *container =
              local_blocks->get_tree_local_blocks(tree_id);
          replace_block(tree_id, csmid, cblk, container);
          cblk_raw = nullptr;
        }
        return p;
      }
      if (!ok) {
        // recycled to a different tree: undo the phantom increment.
        uint64_t gbid = table->get_global_block_offset(cblk);
        free_offset(slice + gbid * 4096);
      }
      cblk_raw = nullptr;
    }

    return malloc_cached_refill(cblk_raw, cbase, csmid, tree_id, alloc_size,
                                size);
  }

#ifdef GALLATIN_CACHED_HOISTED
  // Same as malloc_cached but tree_id + alloc_size are compile-time/cached, so
  // the hot path issues ZERO resolution loads and keeps fewer 64-bit values
  // live. ALLOC_SIZE is a template constant (the fixed slice size), so
  // slice*ALLOC_SIZE folds to an immediate shift and no 8-byte alloc_size lives
  // across the loop -> fewer registers / less spill. Single-thread.
  template <uint64_t ALLOC_SIZE>
  __device__ void *malloc_cached_hoisted(void *&cblk_raw, char *&cbase,
                                         int &csmid, uint16_t tree_id,
                                         uint64_t size) {
    Block *cblk = reinterpret_cast<Block *>(cblk_raw);
    if (cblk != nullptr) {
      uint merged = atomicAdd((unsigned int *)&cblk->malloc_counter, 1u);
      uint slice = merged & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
#ifdef GALLATIN_ARENA
      // Single-size arena: this allocator only ever serves one tree, so a
      // cached block never recycles to a DIFFERENT size -> the tag check is
      // always true. Strip check_valid (the tag compare + branch + rollback)
      // to approach slab's instruction count. ONLY valid for single-size use.
      if (slice < 4096) {
        void *p = cbase + static_cast<uint64_t>(slice) * ALLOC_SIZE;
        if (slice == 4095) {
#ifdef GALLATIN_OUTLINE_ALL
          replace_block_cold(tree_id, csmid, cblk);
#else
          per_size_pinned_blocks *container =
              local_blocks->get_tree_local_blocks(tree_id);
          replace_block(tree_id, csmid, cblk, container);
#endif
          cblk_raw = nullptr;
        }
        return p;
      }
      cblk_raw = nullptr;
#else
      bool ok = cblk->check_valid(merged, tree_id);
      if (ok && slice < 4096) {
        void *p = cbase + static_cast<uint64_t>(slice) * ALLOC_SIZE;
        if (slice == 4095) {
          per_size_pinned_blocks *container =
              local_blocks->get_tree_local_blocks(tree_id);
          replace_block(tree_id, csmid, cblk, container);
          cblk_raw = nullptr;
        }
        return p;
      }
      if (!ok) {  // recycled to another tree: undo the phantom increment
        uint64_t gbid = table->get_global_block_offset(cblk);
        free_offset(slice + gbid * 4096);
      }
      cblk_raw = nullptr;
#endif
    }
    return malloc_cached_refill(cblk_raw, cbase, csmid, tree_id, ALLOC_SIZE,
                                size);
  }
#endif
#undef GALLATIN_REFILL_ATTR
#endif

#ifdef GALLATIN_FLAT_BUFFER
  // O(1) flat-buffer malloc: routes to the pinned-slot array via the
  // __constant__ descriptor table (no local_blocks->block_containers chase).
  // Re-reads the slot every call and tag-validates (never caches the Block*) so
  // cross-size recycling stays safe. Single-thread (call from one lane). Pairs
  // with GALLATIN_CONST_BASE so address translation is constant-based too.
  __device__ void *malloc_flat(uint64_t size) {
    uint16_t tree_id = get_tree_id_from_size(size);
    if (tree_id >= num_trees) return malloc(size);

    Block **slots = gallatin_flat::g_tree_slots[tree_id].blocks;
    uint64_t nblk = gallatin_flat::g_tree_slots[tree_id].num_blocks;
    const uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    int num_attempts = 0;
    while (num_attempts < GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS) {
#ifdef GALLATIN_SPREAD_ATOMIC
      // spread an SM's warps across distinct slots so they cache different
      // blocks -> the reserving atomicAdd distributes instead of funneling to
      // one per-SM block counter. 17 is coprime with pow2 nblk => good spread.
      int base_smid =
          gallatin::utils::get_smid() + (int)((threadIdx.x >> 5) * 17u);
#else
      int base_smid = gallatin::utils::get_smid();
#endif
#ifdef GALLATIN_POW2_BLOCKS
      int smid = base_smid & (int)(nblk - 1);
#else
      int smid = base_smid % nblk;
#endif
      Block *my_block = gallatin::utils::load_acquire(&slots[smid]);
      int probe = 0;
      while (my_block == nullptr && probe < SHARED_BLOCK_COUNTER_CUTOFF) {
#ifdef GALLATIN_POW2_BLOCKS
        smid = (smid + 1) & (int)(nblk - 1);
#else
        smid = (smid + 1) % nblk;
#endif
        my_block = gallatin::utils::load_acquire(&slots[smid]);
        probe++;
      }
      if (my_block == nullptr) {
        if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty()) {
          return malloc(size);
        }
        num_attempts++;
        continue;
      }

      uint64_t global_block_id = table->get_global_block_offset(my_block);
      char *block_base = reinterpret_cast<char *>(
          offset_to_allocation(global_block_id * 4096, tree_id));

      uint merged_count =
          atomicAdd((unsigned int *)&my_block->malloc_counter, 1u);
      uint slice = merged_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);

      if (slice == 4095) {
        // rare (1/4096): navigate to the container only here for replace_block.
        per_size_pinned_blocks *container =
            local_blocks->get_tree_local_blocks(tree_id);
        replace_block(tree_id, smid, my_block, container);
      }

      if (slice < 4096) {
        if (!my_block->check_valid(merged_count, tree_id)) {
          free_offset(slice + global_block_id * 4096);
        } else {
          return block_base + static_cast<uint64_t>(slice) * alloc_size;
        }
      }
      num_attempts++;
    }
    return malloc(size);
  }
#endif

  __device__ void *malloc_single(uint64_t size) {
    uint16_t tree_id = get_tree_id_from_size(size);
    if (tree_id >= num_trees) {
      return malloc(size);
    }

    // Hoist per-allocation invariants OUT of the retry loop (computed once,
    // not on every attempt / not chained behind the atomic).
    per_size_pinned_blocks *local_shared_block_storage =
        local_blocks->get_tree_local_blocks(tree_id);
    const uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    int num_attempts = 0;
    while (num_attempts < GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS) {

      int shared_block_storage_index;
      Block *my_block = local_shared_block_storage->get_valid_block(
          shared_block_storage_index);

      if (my_block == nullptr) {
        if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty()) {
          return malloc(size);  // genuine pressure -> cooperative fallback
        }
        num_attempts++;
        continue;
      }

      uint64_t global_block_id = table->get_global_block_offset(my_block);

      // The block's base address depends only on the block, NOT on the slice we
      // are about to reserve (slice < 4096 never crosses a segment boundary).
      // Compute it BEFORE the atomicAdd so its address-translation loads overlap
      // the atomic's latency; only a single FMA is left on the post-atomic path.
      char *block_base = reinterpret_cast<char *>(
          offset_to_allocation(global_block_id * 4096, tree_id));

      uint merged_count =
          atomicAdd((unsigned int *)&my_block->malloc_counter, 1u);
      uint slice = merged_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);

      if (slice == 4095) {
        replace_block(tree_id, shared_block_storage_index, my_block,
                      local_shared_block_storage);
      }

      if (slice < 4096) {
        if (!my_block->check_valid(merged_count, tree_id)) {
          free_offset(slice + global_block_id * 4096);
        } else {
          return block_base + static_cast<uint64_t>(slice) * alloc_size;
        }
      }

      num_attempts++;
    }

    return malloc(size);
  }

  // General multi-slice slice allocator. Used when a single request needs more
  // than one slice (e.g., 8KB-sized requests on a tree of 4KB slices). The
  // single-slice path above should be preferred for the common case.
  __device__ uint64_t malloc_slice_allocation(uint16_t & tree_id, uint & alloc_count){

     // get local block storage and thread storage
    per_size_pinned_blocks * local_shared_block_storage =
        local_blocks->get_tree_local_blocks(tree_id);

    int shared_block_storage_index;
    Block * my_block;

    int num_attempts = 0;

    // this cycles until we either receive an allocation or fail to request a
    // new block
    while (num_attempts < GALLATIN_MAX_ATTEMPTS) {

    // get_valid_block uses load_acquire, so no fence needed before the read.
    cg::coalesced_group full_warp_team = cg::coalesced_threads();

    cg::coalesced_group coalesced_team = labeled_partition(full_warp_team, tree_id);

    if (coalesced_team.thread_rank() == 0){
      my_block = local_shared_block_storage->get_valid_block(shared_block_storage_index);
    }

    //recoalesce and share block.
    shared_block_storage_index = coalesced_team.shfl(shared_block_storage_index, 0);
    my_block = coalesced_team.shfl(my_block, 0);

    //cycle if we read an old block
    if (my_block == nullptr){
      // Fast bail: no block available and no way to get one
      if (sub_trees[tree_id]->is_empty() && segment_tree->is_empty()){
        return ~0ULL;
      }
      num_attempts+=1;
      continue;
    }

    #if GALLATIN_BLOCK_CHECK
    uint64_t alt_block_segment = table->get_segment_from_block_ptr(my_block);

    uint16_t alt_tree_id = table->read_tree_id(alt_block_segment);

    if (alt_tree_id != tree_id){

      //printf("Trigger bad tree read\n");
      return ~0ULL;
    }
    #endif

    #if GALLATIN_DEBUG_PRINTS


    uint64_t alt_block_segment = table->get_segment_from_block_ptr(my_block);

    uint16_t alt_tree_id = table->read_tree_id(alt_block_segment);

    uint64_t block_id = table->get_global_block_offset(my_block);

    uint64_t relative_block_id = table->get_relative_block_offset(my_block);

    if (tree_id != alt_tree_id){

      if (alt_tree_id > num_trees){

        //this error occurs when the tree id stored differs - this is an indicator of stale pointers

        //stale pointers are now detected and pruned
        printf("Reading from broken segment %llu, alt value %u != %u\n", alt_block_segment, alt_tree_id, tree_id);
        return ~0ULL;

      }



      bool main_tree_owns = sub_trees[tree_id]->query(alt_block_segment);

      bool alt_tree_owns = sub_trees[alt_tree_id]->query(alt_block_segment);

      if (!main_tree_owns){

        if (alt_tree_owns){
          printf("ERROR: Tree %u reading from segment %llu owned by %u\n", tree_id, alt_block_segment, alt_tree_id);
        } else {
          printf("ERROR: Neither %u or %u own segment %llu\n", tree_id, alt_tree_id, alt_block_segment);
        }

      } else {

        if (alt_tree_owns){
          printf("ERR: Trees %u and %u both own segment %llu\n", tree_id, alt_tree_id, alt_block_segment);
        }

      }

      if (!sub_trees[tree_id]->query(alt_block_segment)){
        printf("Sub tree %u does not own segment %llu\n", tree_id, alt_block_segment);
      }
      //this triggers, yeet
      printf("Block %llu: segment %llu relative %llu not init for malloc: %u != %u\n", block_id, alt_block_segment, relative_block_id, tree_id, alt_tree_id);

      __threadfence();

      continue;
    }

    #endif




      // select block to pull from and get global stats
    uint64_t global_block_id = table->get_global_block_offset(my_block);

    //TODO: add check here that global block id does not exceed bounds

    uint group_sum = cg::exclusive_scan(coalesced_team, alloc_count, cg::plus<uint>());


    uint merged_count = my_block->block_malloc_tree_multi_size(coalesced_team, group_sum+alloc_count);

    uint64_t allocation = my_block->extract_count_multi_size(coalesced_team, merged_count, group_sum, alloc_count);

    

    //this is now correct - final allocation may be incorrect, but we need it.
    bool should_replace = (allocation <= 4095 && (allocation + alloc_count) > 4095);

    //leftover is any fragment > 1 that is inside the array region.

    //think this does it?

    // = (allocation+alloc_count > 4095)*(allocation+alloc_count-4096);

    //three cases
    //1 ) entirely valid - alloc_count -1
    //2 ) valid start and invalid_end (4096-allocation)
    //3 ) entirely invalid. - 0.

    bool start_valid = (allocation <= 4095);
    bool end_valid = (allocation+alloc_count <= 4096);

    uint leftover = (start_valid && end_valid)*(alloc_count-1)+(start_valid && (!end_valid))*(4096-allocation);

    my_block->block_correct_frees(coalesced_team, leftover);


    if (allocation + alloc_count > 4096) allocation = ~0ULL;

    should_replace = coalesced_team.ballot(should_replace);

    bool did_replace_block = false;

    if (should_replace){
      if (coalesced_team.thread_rank() == 0){
        did_replace_block = replace_block(tree_id, shared_block_storage_index, my_block, local_shared_block_storage);
      }
    }

    // replace_block uses release-CAS; coalesced_team.sync() below provides the
    // intra-warp ordering for the ballot, no extra fence needed.
    did_replace_block = coalesced_team.ballot(did_replace_block);
    coalesced_team.sync();


    if (allocation != ~0ULL){

      if (!my_block->check_valid(merged_count, tree_id)){


        

        #if GALLATIN_DEBUG_PRINTS

        printf("Gave out wrong offset\n");

        my_block->check_valid(merged_count, tree_id);
        #endif

        // #if GALLATIN_TRAP_ON_ERR
        // asm volatile ("trap;");
        // #endif

        free_offset(allocation+global_block_id*4096);

      } else {
        return allocation + global_block_id*4096;
      }

      

    } else {
      // if (!did_replace_block){
      //   return ~0ULL;
      // }
    }


    num_attempts+=1;

  }

  return ~0ULL;



}


  //replace block with a new one pulled from the system
  //gets called when a block is detected to be empty.
  __device__ bool replace_block(int tree_id, int smid, Block * my_block, per_size_pinned_blocks * my_pinned_blocks){

  	if (my_pinned_blocks->swap_out_block(smid, my_block)){

      // swap_out_block is a release-CAS; no extra fence needed before pulling
      // a fresh block.
  		Block * new_block = request_new_block_from_tree(tree_id);

  		if (new_block == nullptr){

        #if GALLATIN_DEBUG_PRINTS
        printf("Failed to acquire block\n");
        #endif

        //PROCEDURE UPDATE
        //SWAP BLOCK BACK IN
        // this makes 100% space usage a recoverable condition
        // but slows down threads at the OOM threshold
        // if (!my_pinned_blocks->swap_out_nullptr(smid, my_block)){

        //   #if GALLATIN_DEBUG_PRINTS
        //   printf("Incorrect behavior when swapping out block index %d for tree %d\n", smid, tree_id);
        //   #endif

        //   free_block(new_block);

        //   #if GALLATIN_TRAP_ON_ERR
        //   asm volatile ("trap;");
        //   #endif

        // }

  			return false;
  		}

  		if (!my_pinned_blocks->swap_out_nullptr(smid, new_block)){

        #if GALLATIN_DEBUG_PRINTS
  			printf("Incorrect behavior when swapping out block index %d for tree %d\n", smid, tree_id);
        #endif

  			free_block(new_block);

        #if GALLATIN_TRAP_ON_ERR
        asm volatile ("trap;");
        #endif

  			return false;
  		}

  	}

  	return true;


  }


  __device__ uint16_t get_tree_id_from_size(uint64_t size){

      if (size < smallest) return 0;

      return get_first_bit_bigger(size) - smallest_bits;

  }

  // used for poison - return bytes to be allocated
  // for a requested # of bytes needed
  // just adjusts the size and then promotes to next p2
  __device__ uint64_t get_allocated_size(uint64_t bytes_needed){


    if (bytes_needed < 16) bytes_needed = 16;

    //then, determine byte size

    //if moving to segments, determine that.

    int smallest_tree_bits = get_first_bit_bigger(smallest*4096);

    uint16_t bit_size = get_first_bit_bigger(bytes_needed);
    uint16_t tree_size = get_tree_id_from_size(bytes_needed);

    if (tree_size > num_trees){

      int block_tree = (int) bit_size - smallest_tree_bits;

      if (block_tree > num_trees){

        //segment details
        uint64_t alloc_count = (bytes_needed - 1)/ bytes_per_segment + 1;

        return alloc_count*bytes_per_segment;

      }

    }

    return (1ULL << bit_size);

  }

  //v2 of malloc - handle tree_id externally.
  __device__ void * malloc(uint64_t size){

    //updated version for register sharing
    // uint alloc_count = 1;

    // // 0 = slice, 1 = block, 2 = segment
    // int alloc_level = 0;

    // if (size < smallest) size = smallest;

    uint16_t tree_id = get_tree_id_from_size(size);
    uint alloc_count = 1;

    uint64_t offset = ~0ULL;
    uint64_t attempt_counter = 0;

    void * alloc = nullptr;

    if (tree_id >= num_trees){

      int smallest_tree_bits = get_first_bit_bigger(smallest*4096);

      int block_tree = (int) get_first_bit_bigger(size) - smallest_tree_bits;

      if (block_tree < 0){ 


        alloc_count = (1ULL << (tree_id - (num_trees-1)));
        tree_id = num_trees-1;


        //big slice_malloc - fall through

      } else if (block_tree < num_trees){

        //block_malloc;
        //guaranteed safe as block_tree > 0;
        tree_id = (uint16_t) block_tree;


        while (offset == ~0ULL && attempt_counter < GALLATIN_MALLOC_BLOCK_ATTEMPTS){

          offset = malloc_block_allocation(tree_id);
          attempt_counter += 1;

        }

        if (offset != ~0ULL){
           alloc = offset_to_allocation(offset, tree_id);
        }

        return alloc;

      } else {

        //big allocation
        alloc_count = (size - 1)/ bytes_per_segment + 1;
        tree_id = 0;


        while (offset == ~0ULL && attempt_counter < GALLATIN_MALLOC_SEGMENT_ATTEMPTS){

          offset = malloc_segment_allocation(alloc_count);
          attempt_counter +=1;
        
        }

        if (offset != ~0ULL){
           alloc = offset_to_allocation(offset, tree_id);
        }

        return alloc;
      }

    }


    // Common case: a single slice. Route through the specialized fast path.
    if (alloc_count == 1) {
      while (offset == ~0ULL && attempt_counter < GALLATIN_MALLOC_LOOP_ATTEMPTS) {
        offset = malloc_slice_one(tree_id);
        attempt_counter++;
      }
    } else {
      // Multi-slice (sub-block large alloc).
      while (offset == ~0ULL && attempt_counter < GALLATIN_MALLOC_LOOP_ATTEMPTS) {
        offset = malloc_slice_allocation(tree_id, alloc_count);
        attempt_counter++;
      }
    }

    if (offset != ~0ULL){
       alloc = offset_to_allocation(offset, tree_id);
    }

    return alloc;

  }



  __device__ void free(void * allocation){


    //this logic is verifie allocation to offset
    uint64_t segment = table->get_segment_from_ptr(allocation);

    uint16_t tree_id = table->read_tree_id(segment);


    // uint64_t alt_segment = ((uint64_t) allocation - (uint64_t) table->memory)/table->get_bytes_per_segment();

    // if (alt_segment != segment) printf("mismatch: Segment %lu != alt %lu\n", segment, alt_segment);



    //if this is true, removing valid large allocation of unknown size.
    if (tree_id > num_trees && (tree_id != (uint16_t)~0)){

      uint16_t size = tree_id - num_trees - 1;

      // Sanity-check the decoded span before touching segment_tree. A
      // corrupted tree_id (e.g., a use-after-free or a write through a
      // bad pointer) could decode to a size of 0 or one that runs past
      // the allocator's last segment, and silently `return_multiple`'ing
      // that range would corrupt the segment bitmap. Refuse instead.
      if (size == 0 || segment + size > table->num_segments) {
        #if GALLATIN_DEBUG_PRINTS
        printf("free: implausible multi-segment span tree_id=%u -> size=%u "
               "at segment %llu of %llu; refusing\n",
               (unsigned)tree_id, (unsigned)size,
               (unsigned long long)segment,
               (unsigned long long)table->num_segments);
        #endif
        #if GALLATIN_TRAP_ON_ERR
        asm volatile ("trap;");
        #endif
        return;
      }

      // Order matters: clear ownership FIRST, then republish the segment.
      // Both are release-ordered, so any consumer that subsequently observes
      // the segment in segment_tree (via an acquire load of the layer bits)
      // is also guaranteed to observe tree_id == ~0.
      bool reset = table->reset_tree_id(segment, tree_id);

      segment_tree->return_multiple(segment, size);

      #if GALLATIN_DEBUG_PRINTS
      if (!reset){
        printf("Failed to reset tree id for segment %lu\n", segment);
      }
      #endif

      return;
    }


    if (tree_id > num_trees){

      #if GALLATIN_DEBUG_PRINTS

      printf("Tree freeing into uninitialized segment\n");

      #endif


      #if GALLATIN_TRAP_ON_ERR
      asm volatile ("trap;");
      #endif

      return;


    }


    uint64_t offset = allocation_to_offset(allocation, tree_id);

    // uint64_t byte_offset = (char *) allocation - table->memory;

    // uint64_t offset = byte_offset/table->get_tree_alloc_size(tree_id);
   

    #if GALLATIN_DEBUG_PRINTS

      uint64_t raw_bytes = (char *) allocation - table->memory;
    
      uint64_t offset_segment = table->get_segment_from_offset(offset);

      if (segment != offset_segment){
        printf("pointer %llx - bytes: %llu, offset: %llu - Free segment Ids Mismatch: %llu != %llu, tree %u\n", (uint64_t) allocation, raw_bytes, offset, segment, offset_segment, tree_id);
      }

    #endif

    return free_offset(offset);


  }


  // get a new segment for a given tree!
  __device__ int gather_new_segment(uint16_t tree) {

    // request new segment
    uint64_t new_segment_id = segment_tree->malloc_first();

    if (new_segment_id == veb_tree::fail()) {
      // no segment available - this signals allocator full, return nullptr.
      return -1;
    }

    // otherwise, both initialized
    // register segment
    if (!table->setup_segment(new_segment_id, tree)) {
      
      #if GALLATIN_DEBUG_PRINTS
      printf("Failed to acquire updatable segment\n");
      #endif

      //segment_tree->insert_force_update(new_segment_id);
      // abort, but not because no segments are available.
      // this is fine.

      #if GALLATIN_TRAP_ON_ERR
      asm volatile ("trap;");
      #endif

      return -1;
    }

    // setup_segment ends with a release-CAS on chunk_ids; insert_force_update
    // uses release atomicOr at the leaf and propagates upward via more
    // release-ordered RMWs. No additional fences needed for visibility.
    sub_trees[tree]->insert_force_update(new_segment_id);

    return new_segment_id;
  }

  // Per-tree try-lock. acquire-CAS on a dedicated cache line per tree means
  // contention on tree A doesn't bounce tree B's lock line.
  __device__ bool acquire_tree_lock(uint16_t tree) {
    uint expected = 0;
    return gallatin::utils::cas_acquire<uint>(&tree_locks[tree].v, expected,
                                              1u);
  }

  __device__ void release_tree_lock(uint16_t tree) {
    gallatin::utils::store_release<uint>(&tree_locks[tree].v, 0u);
  }


  // gather a new block for a tree.
  // this attempts to pull from a memory segment.
  //  and will atteach a new segment if now
  __device__ Block *request_new_block_from_tree(uint16_t tree) {
    int attempts = 0;

    // Fast bail: if sub-tree and segment tree are both empty, no blocks can be acquired.
    if (sub_trees[tree]->is_empty() && segment_tree->is_empty()) {
      return nullptr;
    }

    while (attempts < REQUEST_BLOCK_MAX_ATTEMPTS) {

      uint64_t segment = sub_trees[tree]->find_random_valid_index();

      if (segment == veb_tree::fail()) {

        // Re-check: if segment tree is empty, no point retrying.
        if (segment_tree->is_empty()) {
          return nullptr;
        }

        if (acquire_tree_lock(tree)) {
          int success = gather_new_segment(tree);
          release_tree_lock(tree);

          if (success == -1) {
            // timeouts should be rare — usually means someone else attached a
            // segment between our find_random and our lock acquire.
            attempts++;
            continue;
          } else {
            segment = success;
          }
        } else {
          continue;
        }
      }



      bool last_block = false;

      // valid segment, get new block.
      Block * new_block = table->get_block(segment, tree, last_block);


      #if GALLATIN_DEBUG_PRINTS

      //verify segments match

      if (new_block != nullptr){

        uint64_t block_segment = table->get_segment_from_block_ptr(new_block);

        if (block_segment != segment){

           printf("Segment misaligned when requesting block: %llu != %llu\n", block_segment, segment);
        }

       

      }

      #endif

      if (last_block) {
        // if the segment is fully allocated, it can be detached
        // and returned to the segment tree when empty
        

        bool removed = sub_trees[tree]->remove(segment);


        #if GALLATIN_DEBUG_PRINTS

        //only worth bringing up if it failed.
        if (!removed){
          printf("Failed Remove segment %llu from tree %u: %d success?\n", segment, tree, removed);
        }
        #endif

        if (acquire_tree_lock(tree)) {
          gather_new_segment(tree);
          release_tree_lock(tree);
        }
      }

      // if (!valid){
      //   free_block(new_block);
      //   new_block = nullptr;
      // }

      if (new_block != nullptr) {
        return new_block;
      }

      // get_block rejected a segment that find_random_valid_index reported as
      // valid: the segment has no free block right now (active_counts
      // exhausted) yet is still listed in sub_trees[tree]. Without doing
      // anything here, find_random keeps re-selecting that same dead segment
      // and we spin forever (this path does not increment `attempts`).
      //
      // This is reachable for the largest tree, where a segment holds exactly
      // one block (bytes_per_segment == tree alloc size * 4096): once that
      // block is taken — e.g. pre-consumed by the boot wavefront — the segment
      // is exhausted but remains listed, violating the invariant "a segment in
      // sub_trees[tree] has a free block of tree `tree`."
      //
      // Evict it from the sub-tree (a freed block re-inserts the segment) and
      // count an attempt so the loop is bounded and falls through to
      // gather_new_segment for a fresh segment.
      sub_trees[tree]->remove(segment);
      __threadfence();
      attempts++;
    }

    // on attempt failures, allocator is full
    return nullptr;
  }


  //called after memory is freed.
  //this helper separates the logic between acquiring a block and returning
  //so that the free can cleanly acquire size before proceeding to free.
  __device__ void return_block(Block * block_to_free, uint64_t segment, uint16_t tree){


    uint64_t num_blocks = table->get_blocks_per_segment(tree);


    uint reserved_slot = table->reserve_segment_slot(block_to_free, segment, tree, num_blocks);


    if (1.0*reserved_slot/num_blocks >= REREGISTER_CUTOFF && ((1.0*(reserved_slot-1)/num_blocks) < REREGISTER_CUTOFF)){
      // re-publish to sub-tree; insert_force_update is release-ordered.
      sub_trees[tree]->insert_force_update(segment);
    }

    bool need_to_deregister = table->finish_freeing_block(segment, num_blocks);

    if (need_to_deregister) {

      #if DEBUG_NO_FREE
      return;
      #endif

      // Order is critical:
      //   1. remove from sub_tree (release atomicAnd)
      //   2. reset_tree_id back to ~0 (release CAS)
      //   3. publish back to segment_tree (release atomicOr)
      // Any consumer that subsequently finds the segment in segment_tree (via
      // an acquire load) will observe tree_id == ~0 and an empty sub_tree slot.
      sub_trees[tree]->remove(segment);

      if (!table->reset_tree_id(segment, tree)){

        #if GALLATIN_DEBUG_PRINTS
        printf("Failed to reset tree id for segment %llu, old ID %u\n", segment, tree);
        #endif

        #if GALLATIN_TRAP_ON_ERR
        asm volatile ("trap;");
        #endif
      }

      if (!segment_tree->insert_force_update(segment)){

        #if GALLATIN_DEBUG_PRINTS

        printf("Failed to reinsert segment %llu into segment tree\n", segment);
        #endif

      }
    }


  }



  // return a block to the system
  // this is called by a block once all allocations have been returned.
  __device__ void free_block(Block *block_to_free) {

    uint64_t segment = table->get_segment_from_block_ptr(block_to_free);

    uint16_t tree = table->read_tree_id(segment);

    return_block(block_to_free, segment, tree);
  }



  // return a uint64_t to the system
  //fuckk this doesn't work.
  //needs to be a system variable.

  __device__ void free_offset(uint64_t malloc) {

    // get block
    uint64_t block_id = malloc/4096;


    #if GALLATIN_TEAM_FREE

      cg::coalesced_group full_warp_team = cg::coalesced_threads();

      cg::coalesced_group coalesced_team = labeled_partition(full_warp_team, block_id);

      Block * my_block = table->get_block_from_global_block_id(block_id);

      if (coalesced_team.thread_rank() == 0){

        if (my_block->block_free_multiple(coalesced_team.size())){

            #if !DEBUG_NO_FREE
            my_block->reset_free();
            #endif

            free_block(my_block);
        }
      }

    #else

      Block * my_block = table->get_block_from_global_block_id(block_id);

      if (my_block->block_free()){


        #if !DEBUG_NO_FREE
        my_block->reset_free();
        #endif

        free_block(my_block);

        

      }

    #endif

    return;
  }


  //given a uint64_t allocation, return a void * corresponding to the desired memory
  __device__ void * offset_to_allocation(uint64_t offset, uint16_t tree_id){


      //to start, get the segment

      return table->offset_to_allocation(offset, tree_id);

  }


  //given a void * and the known size (expressed as tree id), translate to offset in global space.
  __device__ uint64_t allocation_to_offset(void * allocation, uint16_t tree_id){

    
      return table->allocation_to_offset(allocation, tree_id);
    
  }




  // print useful allocator info.
  // this returns the number of segments owned by each tree
  // and maybe more useful things later.
  __host__ void print_info() {
    my_type *host_version = copy_to_host<my_type>(this);

    uint64_t segments_available = host_version->segment_tree->report_fill();

    uint64_t max_segments = host_version->segment_tree->report_max();


    printf("\n\033[1;32mGallatin usage stats:\033[1;0m\n");
    printf("Segment tree sees %lu/%lu segments available\n", segments_available,
           max_segments);

    sub_tree_type **host_trees = copy_to_host<sub_tree_type *>(
        host_version->sub_trees, host_version->num_trees);

    for (int i = 0; i < host_version->num_trees; i++) {
      uint64_t sub_segments = host_trees[i]->report_fill();

      uint64_t sub_max = host_trees[i]->report_max();

      printf("Tree %d: size %lu, owns %lu/%lu\n", i, table->get_tree_alloc_size(i), sub_segments, sub_max);
    }

    // uint64_t free_indices = host_version->table->report_free();

    // printf("Table reports %llu indices have been freed\n", free_indices);

    // uint64_t live_indices = host_version->table->report_live();

    // printf("Table reports %llu indices have been used\n", live_indices);

    cudaFreeHost(host_trees);

    cudaFreeHost(host_version);

    printf("\n\033[1;32mTree usage stats:\033[1;0m\n");

    this->print_usage();

    this->print_overhead();


  }

  static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree) {
    return alloc_table<bytes_per_segment, smallest>::get_blocks_per_segment(
        tree);
  }

  //return maximum # of possible allocations per segment.
  static __host__ __device__ uint64_t get_max_allocations_per_segment(){

    return alloc_table<bytes_per_segment, smallest>::get_max_allocations_per_segment();

  }

  //launch a thread to calculate overhead
  __device__ uint64_t calculate_overhead(){

    uint64_t overhead = 0;

    overhead += sizeof(my_type);

    

    //segment tree
    overhead += segment_tree->calculate_overhead();

    //sub trees

    for (int i =0; i < num_trees; i++){
      overhead += sub_trees[i]->calculate_overhead();
    }

    //local blocks

    for (int i = 0; i < num_trees; i++){

      overhead += local_blocks->get_tree_local_blocks(i)->calculate_overhead();

    }

    //mem table

    overhead += table->calculate_overhead();

    return overhead;


  }

  __host__ void print_overhead(){


    print_overhead_kernel<my_type><<<1,1>>>(this);

    cudaDeviceSynchronize();



  }


  __host__ void print_usage(){

    my_type *host_version = copy_to_host<my_type>(this);


    for (uint16_t i = 0; i < host_version->num_trees; i++){

      print_guided_fill_host(i);

    }


    cudaFreeHost(host_version);


  }

  //generate average fill using the info from the segment tree
  __device__ void print_guided_fill(uint16_t id){


    uint64_t count = 0;

    int malloc_count = 0;
    int free_count = 0;

    uint64_t nblocks = table->get_blocks_per_segment(id);

    for (uint64_t i = 0; i < table->num_segments; i++){
    

      if (table->read_tree_id(i) == id){
      //if (sub_trees[id]->query(i)){


        if (table->active_counts[i] == -1) continue;

        if (table->active_counts[i] > nblocks){
          printf("Big value: index %lu has %d > %lu\n", i, table->active_counts[i], nblocks);
        }
        count += 1;
        malloc_count += table->active_counts[i];
        free_count += nblocks - table->active_counts[i];

      }


  }


  printf("Tree %u: %lu live blocks | avg available %f / %llu | avg in use %f / %llu\n", id, count, 1.0*malloc_count/count, nblocks, 1.0*free_count/count, nblocks);


  }


  __host__ void print_guided_fill_host(uint16_t id){

    print_guided_fill_kernel<my_type><<<1,1>>>(this, id);

  }

  __host__ void print_segment_fills(){

    print_segment_fill_kernel<my_type><<<2000, 256>>>(this);

  }

  // Return the number of bytes currently allocated from a given tree.
  // Safe to call only when no allocations/frees are in flight.
  __device__ uint64_t get_tree_bytes_in_use(uint16_t tree_id){

    uint64_t alloc_size = table->get_tree_alloc_size(tree_id);
    uint64_t nblocks = table->get_blocks_per_segment(tree_id);
    uint64_t total = 0;

    for (uint64_t seg = 0; seg < table->num_segments; seg++){

      if (table->read_tree_id(seg) != tree_id) continue;

      uint64_t base = seg * table->blocks_per_segment;

      for (uint64_t b = 0; b < nblocks; b++){

        Block * blk = table->get_block_from_global_block_id(base + b);
        uint malloced = blk->clip_count(blk->malloc_counter);
        uint freed = blk->free_counter;

        if (malloced > freed){
          total += (malloced - freed) * alloc_size;
        }
      }
    }

    return total;
  }

  //returns true if this allocation is inside the range of the allocator
  __device__ bool owns_allocation(void * alloc){

    return table->owns_allocation(alloc);

  }



};

}  // namespace allocators

}  // namespace Gallatin

#endif  // End of VEB guard