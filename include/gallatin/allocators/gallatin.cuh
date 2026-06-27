#ifndef GALLATIN_ALLOCATOR
#define GALLATIN_ALLOCATOR

// Cold-path outlining (opt #1, DEFAULT ON): the rare allocation paths
// (block/segment alloc, block replace/recycle) are __noinline__ so they leave
// the hot malloc/free kernel's register footprint -> higher occupancy.
// Measured +11-35% on chain/cuckoo insert (validated). Opt out with
// -DGALLATIN_NO_OUTLINE_COLD to A/B.
#ifndef GALLATIN_NO_OUTLINE_COLD
#define GALLATIN_COLD_ATTR __noinline__
#else
#define GALLATIN_COLD_ATTR
#endif
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
// Overridable from the compiler (-DMIN_PINNED_CUTOFF=N) so the boot-wavefront
// segment budget can be swept without editing source. Default stays 32.
#ifndef MIN_PINNED_CUTOFF
#define MIN_PINNED_CUTOFF 32
#endif
// Geometric START of the per-tree pinned wavefront (tree 0 gets this many slots,
// halving per larger tree down to MIN_PINNED_CUTOFF, capped by the pool budget).
// Raising the START (not the MIN floor) spreads warps across more counters on the
// hot small-slice trees -- the contention lever -- while the halving keeps large-
// slice trees' pinned memory bounded. -DGALLATIN_PINNED_WAVEFRONT=N to sweep.
#ifndef GALLATIN_PINNED_WAVEFRONT
#define GALLATIN_PINNED_WAVEFRONT 256
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
template <typename table_t>
static __global__ void boot_segments_deterministic(
    veb_tree *segment_tree,
    veb_tree **sub_trees,
    table_t *table,
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

  // Pre-claim boot segments from segment_tree, initialize their metadata, and
  // publish them into the owning tree's sub_tree. Single thread per segment,
  // deterministic addresses — no contention beyond the natural per-64bit-word
  // atomic. setup_segment runs here (not per slot) so that all of a segment's
  // metadata + tree_id are published before boot_shared_block_container's slot
  // threads call get_block — the kernel boundary between the two launches is
  // the barrier that lets multiple slots safely pack onto one segment.
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

    if (!table->setup_segment(tid, owning_tree)) {
      // chunk_ids is ~0 from table init, so the release-CAS must win. A failure
      // is a hard invariant violation (duplicate segment id).
      printf("Boot (deterministic): setup_segment failed for segment %llu (tree %d)\n",
             (unsigned long long)tid, owning_tree);
      #if GALLATIN_TRAP_ON_ERR
      asm volatile("trap;");
      #endif
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
      // Pack blocks_per_segment slots onto each boot segment: slot `tid` pulls
      // a block from segment offset[tree] + tid/blocks_per_segment. Several
      // slots therefore share a segment and each get a distinct block via
      // get_block's atomic active_counts reservation.
      uint64_t blocks_per_seg = alloc->table->get_blocks_per_segment(tree_id);
      uint64_t segment_id =
          boot_segment_offsets[tree_id] + (tid / blocks_per_seg);
      alloc->boot_block_deterministic(tree_id, (int)tid, segment_id);
    }
  }
}


// ---- Block-pointer cache (resident fast path over the REAL Block::malloc_counter) ----
// Caches one Block* per (tree, slot) so the hot allocation path skips
// get_valid_block's probe + the tree-local-blocks chain. Allocation STILL flows
// through Block::malloc_counter (single source of truth), reusing the proven
// block_malloc / check_valid / replace_block / free_offset machinery -- so the
// shadow-counter bug classes (count overflow, generation wraparound, torn base,
// dual-counter double-alloc, leak) cannot occur. Memory-neutral: one pointer per
// already-pinned slot; no extra pinned blocks. Disable with -DGALLATIN_NO_BLOCK_CACHE.
#ifndef GALLATIN_NO_BLOCK_CACHE
#define GALLATIN_BLOCK_CACHE
#endif

#ifdef GALLATIN_BLOCK_CACHE
namespace block_cache {
static constexpr int MAX_TREES = 16;
#ifndef GALLATIN_CACHE_MAX_N
#define GALLATIN_CACHE_MAX_N 4096
#endif
static constexpr int MAX_N = GALLATIN_CACHE_MAX_N;  // must cover per-tree num_blocks
__device__ Block *g_block[MAX_TREES * MAX_N];
__device__ int g_nblk[MAX_TREES];
// fmix32: thread-id -> slot. Hashing the thread id (not warp id) spreads a warp's
// lanes across distinct slots/blocks, so per-lane reserves hit distinct counters
// (low contention) without pinning extra blocks.
__device__ __forceinline__ unsigned int slot_hash(unsigned int x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

#ifdef GALLATIN_STATIC_COUNTER
// Off-block padded reservation counter (one atomic, not gated on a block load).
// g_ctr64[slot*CSTRIDE64] packs (gen<<32)|count: 32-bit count can't overflow (R2),
// 32-bit gen can't wrap (R3). CSTRIDE64=16 -> one 64-bit counter per 128B cacheline
// (kills dense-array false sharing). g_sbase = slice-0 base; g_prev = small ring of
// retired blocks by gen (lets a torn reservation roll back on the correct block, R1).
static constexpr int CSTRIDE64 = 16;
static constexpr int RING = 8;
__device__ unsigned long long g_ctr64[MAX_TREES * MAX_N * CSTRIDE64];
__device__ uint64_t g_sbase[MAX_TREES * MAX_N];
__device__ Block *g_prev[MAX_TREES * MAX_N * RING];
#endif
}  // namespace block_cache

template <typename allocator>
__global__ void block_cache_fill_kernel(allocator *alloc) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nt = alloc->num_trees;
  if (nt > block_cache::MAX_TREES) nt = block_cache::MAX_TREES;
  for (int t = 0; t < nt; t++) {
    auto *st = alloc->local_blocks->get_tree_local_blocks(t);
    int n = (int)st->num_blocks;
    if (n > block_cache::MAX_N) n = block_cache::MAX_N;
    block_cache::g_nblk[t] = n;
    for (int s = 0; s < n; s++) {
      int sidx = t * block_cache::MAX_N + s;
      Block *b = st->blocks[s];
      block_cache::g_block[sidx] = b;
#ifdef GALLATIN_STATIC_COUNTER
      block_cache::g_ctr64[sidx * block_cache::CSTRIDE64] = 0ULL;  // gen 0, count 0
      if (b != nullptr) {
        uint64_t gbid = alloc->table->get_global_block_offset(b);
        block_cache::g_sbase[sidx] =
            (uint64_t)alloc->offset_to_allocation(gbid * 4096, (uint16_t)t);
      } else {
        block_cache::g_sbase[sidx] = 0;
        // empty slot -> stamp full so reservers probe past it until a swap fills it
        block_cache::g_ctr64[sidx * block_cache::CSTRIDE64] = 4096ULL;
      }
      for (int r = 0; r < block_cache::RING; r++)
        block_cache::g_prev[sidx * block_cache::RING + r] = nullptr;
#endif
    }
  }
}

template <typename allocator>
__host__ inline void block_cache_publish(allocator *dev_alloc) {
  block_cache_fill_kernel<allocator><<<1, 1>>>(dev_alloc);
  GPUErrorCheck(cudaDeviceSynchronize());
}
#endif


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

    // Per-tree wavefront sizing. The wavefront keeps `slots` pinned blocks hot
    // per tree (geometric recipe: 128, halve per tree, floored at
    // MIN_PINNED_CUTOFF, capped at 1/WAVEFRONT_BUDGET_FRACTION of the tree's
    // blocks). Those blocks come from segments the tree owns — and CRITICALLY
    // a single segment yields blocks_per_segment[t] blocks, so `slots` blocks
    // need only ceil(slots / blocks_per_segment[t]) SEGMENTS, not one segment
    // per slot.
    //
    // The deterministic boot used to grab one whole segment per slot. For
    // small-slice trees (blocks_per_segment up to 256) that withheld
    // (slots - segs) extra segments per tree from the pool — on a 9-tree /
    // 800-segment config it pinned 416 segments instead of ~66, starving any
    // single-size workload (a size-16 run wanting 90% of the pool missed
    // ~26%). We now pack blocks_per_segment slots into each boot segment and
    // grant only the segments actually needed.
    constexpr uint64_t WAVEFRONT_BUDGET_FRACTION = 4;
    uint16_t *tree_slot_counts =
        gallatin::utils::get_host_version<uint16_t>(num_trees);
    uint64_t *tree_segment_counts =
        gallatin::utils::get_host_version<uint64_t>(num_trees);

    uint64_t blocks_per_pinned_block = GALLATIN_PINNED_WAVEFRONT;
    uint64_t geom = blocks_per_pinned_block;
    bool any_reduced = false;
    // Boot segments are removed from segment_tree and dedicated to a tree, so
    // the running total must fit max_chunks. Track remaining pool capacity in
    // SEGMENTS and reserve one per still-unsized tree so every tree can boot
    // at least one slot on small pools.
    uint64_t remaining_pool = max_chunks;
    for (uint16_t t = 0; t < num_trees; ++t) {
      uint64_t blocks_per_seg =
          alloc_table<bytes_per_segment, smallest>::get_blocks_per_segment(t);

      // Wavefront slot target.
      uint64_t target =
          geom < MIN_PINNED_CUTOFF ? (uint64_t)MIN_PINNED_CUTOFF : geom;
      uint64_t slot_cap =
          (max_chunks * blocks_per_seg) / WAVEFRONT_BUDGET_FRACTION;
      if (slot_cap == 0) slot_cap = 1;  // always at least one slot
      if (target > slot_cap) target = slot_cap;

      // Segments needed to back `target` slots (pack blocks_per_seg per segment).
      uint64_t segs_needed = (target + blocks_per_seg - 1) / blocks_per_seg;

      // Clamp granted segments to the remaining pool, reserving 1 per
      // still-unsized tree. A tree clamped to 0 segments simply boots with an
      // empty wavefront (get_valid_block probes forward at first use).
      uint64_t trees_left = num_trees - t;
      uint64_t reserve_for_others = trees_left > 1 ? (trees_left - 1) : 0;
      uint64_t cap_segs = remaining_pool > reserve_for_others
                              ? (remaining_pool - reserve_for_others)
                              : 0;
      if (segs_needed > cap_segs) segs_needed = cap_segs;

      // Slots can't exceed what the granted segments can back.
      uint64_t actual_slots = segs_needed * blocks_per_seg;
      if (actual_slots > target) actual_slots = target;

      if (actual_slots < target) any_reduced = true;
      tree_slot_counts[t] = (uint16_t)actual_slots;
      tree_segment_counts[t] = segs_needed;
      remaining_pool -= segs_needed;

      if (print_info && actual_slots < target) {
        fprintf(stderr,
                "gallatin: tree %u (slice %llu B) wavefront reduced to %llu "
                "slots / %llu segments (remaining pool %llu)\n",
                (unsigned)t,
                (unsigned long long)
                    alloc_table<bytes_per_segment, smallest>::get_tree_alloc_size(t),
                (unsigned long long)actual_slots,
                (unsigned long long)segs_needed,
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

    // Deterministic boot setup: compute prefix-sum of per-tree SEGMENT counts
    // so tree t owns boot segments [offset[t], offset[t+1]). boot_segments_
    // deterministic uses this to (a) clear sub_trees as before, (b) clear
    // the [0, total_boot_segments) range of segment_tree (those segments are
    // now owned by trees), (c) setup_segment each boot segment (init metadata
    // + publish tree_id), and (d) insert it into its owning sub_tree. The
    // per-slot boot then packs blocks_per_segment slots into each segment.
    uint64_t *boot_segment_offsets_host =
        gallatin::utils::get_host_version<uint64_t>(num_trees + 1);
    boot_segment_offsets_host[0] = 0;
    for (uint16_t t = 0; t < num_trees; ++t) {
      boot_segment_offsets_host[t + 1] =
          boot_segment_offsets_host[t] + tree_segment_counts[t];
    }
    uint64_t total_boot_segments = boot_segment_offsets_host[num_trees];
    cudaFreeHost(tree_segment_counts);
    // Invariant: the sizing loop clamps granted segments to remaining pool
    // capacity, so the prefix sum is always <= max_chunks. Defensive assert
    // in debug builds; release builds skip the check.
    assert(total_boot_segments <= max_chunks);
    uint64_t *boot_segment_offsets_dev =
        gallatin::utils::move_to_device<uint64_t>(
            boot_segment_offsets_host, num_trees + 1);

    // Table must exist before boot_segments_deterministic so that kernel can
    // setup_segment (init per-segment metadata + publish tree_id) under the
    // kernel-boundary barrier, before any slot thread calls get_block on it.
    host_version->table =
        alloc_table<bytes_per_segment, smallest>::generate_on_device_nowait(
            max_bytes, memory_control);

    boot_segments_deterministic<<<(max_chunks - 1) / 512 + 1, 512>>>(
        host_version->segment_tree, host_version->sub_trees,
        host_version->table, max_chunks, num_trees, total_boot_segments,
        boot_segment_offsets_dev);

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

#ifdef GALLATIN_BLOCK_CACHE
    // Publish the pinned blocks into the block-pointer cache (after boot has
    // populated every per-tree wavefront slot).
    block_cache_publish<my_type>(device_version);
#endif

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
  // already computed `segment_id` (offset[tree] + slot/blocks_per_segment), and
  // boot_segments_deterministic has already cleared this segment from
  // segment_tree, run setup_segment on it, and inserted it into
  // sub_trees[tree_id]. Multiple slot threads may share a segment_id; each
  // pulls a distinct block via get_block's atomic active_counts reservation.
  // We just pull a block and publish it into the per-tree pinned slot.
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

    // If this boot grab consumed the segment's last free block, the segment is
    // now exhausted. This happens when blocks_per_segment slots packed onto one
    // segment (so all its blocks are pinned) or for the largest tree where
    // blocks_per_segment == 1. boot_segments_deterministic published it into
    // sub_trees[tree], so we detach it here, mirroring the runtime last_block
    // path in request_new_block_from_tree. Otherwise the segment stays listed
    // with no free block and the first runtime request livelocks re-selecting
    // it. A partially-filled boot segment (the remainder of a tree, < bps slots)
    // is NOT exhausted and correctly stays listed with its free blocks.
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


  GALLATIN_COLD_ATTR
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
  
  GALLATIN_COLD_ATTR
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
  GALLATIN_COLD_ATTR
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

#ifdef GALLATIN_BLOCK_CACHE
  // Retire a full cached block: reuse the proven single-winner replace_block
  // (only the count==4095 thread calls this), then refresh the cache slot to the
  // fresh block replace_block published into the pinned pool.
  __device__ void replace_cached(uint16_t tree_id, int idx, int slot, Block *old_block) {
    per_size_pinned_blocks *container = local_blocks->get_tree_local_blocks(tree_id);
    if (replace_block(tree_id, slot, old_block, container)) {
      Block *fresh = gallatin::utils::load_acquire(&container->blocks[slot]);
      gallatin::utils::store_release(&block_cache::g_block[idx], fresh);
    }
  }

  // Self-contained cached slice allocator for a static-managed tree. Coalesces the
  // warp's lanes that want this tree (labeled_partition) so the GROUP reserves a
  // contiguous run with ONE atomicAdd on the resident block's REAL malloc_counter
  // (the faithful Gallatin coalescing -- critical for tile-16, where two tiles
  // share a warp). Mirrors malloc_slice_one exactly, but the block comes from the
  // cache (g_block[idx]) instead of get_valid_block's probe; the count==4095 lane
  // is the SOLE replacer (replace_cached, which refreshes the cache slot). Linear-
  // probes slots via num_attempts. Returns nullptr only on genuine exhaustion.
  __device__ void *malloc_cached(uint16_t tree_id) {
    int nblk = block_cache::g_nblk[tree_id];
    unsigned int slot0 =
        block_cache::slot_hash(blockIdx.x * blockDim.x + threadIdx.x);
    int max_attempts = GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS;
    int num_attempts = 0;
    while (num_attempts < max_attempts) {
      cg::coalesced_group full_warp_team = cg::coalesced_threads();
      cg::coalesced_group team = labeled_partition(full_warp_team, tree_id);

      // Leader picks the (team-uniform) slot, loads the cached block, and
      // overflow-pre-checks it; broadcast so the whole team acts on one block.
      int slot = 0, idx = 0;
      Block *block = nullptr;
      unsigned int precount = 4096;
      if (team.thread_rank() == 0) {
        slot = (int)((slot0 + (unsigned int)num_attempts) % (unsigned int)nblk);
        idx = tree_id * block_cache::MAX_N + slot;
        block = gallatin::utils::load_acquire(&block_cache::g_block[idx]);
        if (block != nullptr)
          precount = gallatin::utils::load_acquire(
                         (unsigned int *)&block->malloc_counter) &
                     BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      }
      idx = team.shfl(idx, 0);
      slot = team.shfl(slot, 0);
      block = team.shfl(block, 0);
      precount = team.shfl(precount, 0);
      // Skip an empty or already-full slot (overflow guard: never atomicAdd a full
      // block -> count can't climb past 2^20 into the tag bits).
      if (block == nullptr || precount >= 4096) { num_attempts++; continue; }

      uint64_t global_block_id = table->get_global_block_offset(block);
      uint merged_count = block->block_malloc_tree(team);  // ONE coalesced atomicAdd
      uint true_count = merged_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);
      uint64_t allocation = true_count + team.thread_rank();
      bool valid = allocation < 4096;
      if (allocation == 4095)
        replace_cached(tree_id, idx, slot, block);  // sole replacer per batch
      team.sync();
      if (valid) {
        if (block->check_valid(merged_count, tree_id))
          return offset_to_allocation(allocation + global_block_id * 4096, tree_id);
        free_offset(allocation + global_block_id * 4096);  // ABA -> roll back
      }
      num_attempts++;
    }
    return nullptr;  // genuine exhaustion -> caller records a miss
  }
#endif

#ifdef GALLATIN_STATIC_COUNTER
  // Retire a full static slot (sole caller = the count==4095 lane, single-winner).
  // Push the old block into the per-gen ring so a torn gen-G reservation can still
  // roll back on it (R1), then pull a fresh block via the proven replace_block,
  // publish base/block, and bump gen with count reset (release via atomicExch).
  __device__ void swap_slot_static(uint16_t tree_id, int sidx, int slot, unsigned int gen) {
    per_size_pinned_blocks *container = local_blocks->get_tree_local_blocks(tree_id);
    Block *old_block = block_cache::g_block[sidx];
    block_cache::g_prev[sidx * block_cache::RING + (gen % block_cache::RING)] = old_block;
    __threadfence();
    if (replace_block(tree_id, slot, old_block, container)) {
      Block *fresh = gallatin::utils::load_acquire(&container->blocks[slot]);
      block_cache::g_block[sidx] = fresh;
      uint64_t gbid = table->get_global_block_offset(fresh);
      gallatin::utils::store_release(&block_cache::g_sbase[sidx],
          (uint64_t)offset_to_allocation(gbid * 4096, tree_id));
      __threadfence();  // publish base/block before the gen bump
      atomicExch(&block_cache::g_ctr64[sidx * block_cache::CSTRIDE64],
                 ((unsigned long long)(gen + 1)) << 32);          // new gen, count 0
    } else {
      block_cache::g_block[sidx] = nullptr;
      atomicExch(&block_cache::g_ctr64[sidx * block_cache::CSTRIDE64],
                 (((unsigned long long)(gen + 1)) << 32) | 4096ULL);  // full -> reservers probe past
    }
  }

  // One-atomic static fast path. The reserving atomicAdd hits a COMPUTED padded
  // address (g_ctr64[slot]) with NO preceding dependent load -- unlike the cache's
  // block-gated atomicAdd -- which is the latency win. Coalesced (tile-16 safe).
  // All races handled: R1 torn base (read base after reserve + gen recheck + ring
  // rollback), R2/R3 (64-bit), R4/R7 (count==4095 sole swapper), R5 (probe past
  // full), R6 (rollback keeps free_counter exact), R9 (no cooperative fallback).
  __device__ void *malloc_static(uint16_t tree_id) {
    int nblk = block_cache::g_nblk[tree_id];
    unsigned int slot0 =
        block_cache::slot_hash(blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t alloc_size = table->get_tree_alloc_size(tree_id);
    int max_attempts = GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS;
    int num_attempts = 0;
    while (num_attempts < max_attempts) {
      cg::coalesced_group full_warp_team = cg::coalesced_threads();
      cg::coalesced_group team = labeled_partition(full_warp_team, tree_id);

      int slot = 0, sidx = 0;
      unsigned long long merged = 0;
      if (team.thread_rank() == 0) {
        slot = (int)((slot0 + (unsigned int)num_attempts) % (unsigned int)nblk);
        sidx = tree_id * block_cache::MAX_N + slot;
        merged = atomicAdd(&block_cache::g_ctr64[sidx * block_cache::CSTRIDE64],
                           (unsigned long long)team.size());  // THE one atomic
      }
      slot = team.shfl(slot, 0);
      sidx = team.shfl(sidx, 0);
      merged = team.shfl(merged, 0);
      unsigned int gen = (unsigned int)(merged >> 32);
      unsigned int count = (unsigned int)(merged & 0xffffffffu) + team.thread_rank();
      if (count >= 4096) { num_attempts++; continue; }  // full -> probe (64-bit: no overflow)

      uint64_t sbase = gallatin::utils::load_acquire(&block_cache::g_sbase[sidx]);  // AFTER reserve
      unsigned int gen_now = (unsigned int)(gallatin::utils::load_acquire(
          &block_cache::g_ctr64[sidx * block_cache::CSTRIDE64]) >> 32);
      if (gen_now != gen) {
        // R1: a swap raced during the base load. Roll back this slice on gen's
        // block (recovered from the ring) so free_counter still reaches 4096.
        Block *bg = block_cache::g_prev[sidx * block_cache::RING + (gen % block_cache::RING)];
        if (bg != nullptr)
          free_offset(table->get_global_block_offset(bg) * 4096 + count);
        num_attempts++;
        continue;
      }
      if (count == 4095) swap_slot_static(tree_id, sidx, slot, gen);  // sole swapper
      return (void *)(sbase + (uint64_t)count * alloc_size);
    }
    return nullptr;  // genuine exhaustion
  }

  // ---- Stateful (per-thread context) fast path ----
  // The caller (device_allocator_context) caches {cidx, cbase, cgen} across the
  // insert loop. The hot path is then ONE atomicAdd on the cached padded counter
  // with NO base load and NO gen revalidate: the atomicAdd's gen result, compared
  // to the cached cgen, validates that the slot has NOT swapped -- which (since a
  // swap is the only thing that moves g_base and it bumps gen) guarantees the
  // cached cbase still belongs to this generation. A mismatch (swapped) or a full
  // slot just misses -> caller drops into gstatic_slow (the safe re-resolve). This
  // is the safe analogue of the broken a792 fastslow: same one-atomic hot path,
  // but cbase/cgen are register-resident so no torn base read can occur.
  __device__ void *gstatic_fast(int cidx, uint64_t cbase, unsigned int cgen,
                                uint64_t alloc_size) {
    if (cidx < 0) return nullptr;
    unsigned long long merged = atomicAdd(
        &block_cache::g_ctr64[cidx * block_cache::CSTRIDE64], 1ULL);
    unsigned int gen = (unsigned int)(merged >> 32);
    unsigned int count = (unsigned int)(merged & 0xffffffffu);
    if (gen == cgen && count < 4096) {
      if (count == 4095)
        swap_slot_static((uint16_t)(cidx / block_cache::MAX_N), cidx,
                         cidx % block_cache::MAX_N, gen);
      return (void *)(cbase + (uint64_t)count * alloc_size);
    }
    return nullptr;  // swapped or full -> caller re-resolves via gstatic_slow
  }

  // Safe re-resolve: like malloc_static (single thread) but also returns the slot
  // to cache. Handles R1 (read base after reserve + gen recheck + ring rollback),
  // R5 (probe past full), R6 (rollback keeps free_counter exact). On the slice that
  // fills a slot (count==4095) it swaps and returns cidx=-1 so the caller does NOT
  // cache a just-swapped slot.
  __device__ void *gstatic_slow(uint16_t tree_id, uint64_t alloc_size,
                                int &o_cidx, uint64_t &o_cbase, unsigned int &o_cgen) {
    o_cidx = -1; o_cbase = 0; o_cgen = 0;
    int nblk = block_cache::g_nblk[tree_id];
    unsigned int slot0 =
        block_cache::slot_hash(blockIdx.x * blockDim.x + threadIdx.x);
    int max_attempts = GALLATIN_MAX_ATTEMPTS * GALLATIN_MALLOC_LOOP_ATTEMPTS;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
      int slot = (int)((slot0 + (unsigned int)attempt) % (unsigned int)nblk);
      int sidx = tree_id * block_cache::MAX_N + slot;
      unsigned long long merged = atomicAdd(
          &block_cache::g_ctr64[sidx * block_cache::CSTRIDE64], 1ULL);
      unsigned int gen = (unsigned int)(merged >> 32);
      unsigned int count = (unsigned int)(merged & 0xffffffffu);
      if (count >= 4096) continue;
      uint64_t sbase = gallatin::utils::load_acquire(&block_cache::g_sbase[sidx]);
      if ((unsigned int)(gallatin::utils::load_acquire(
              &block_cache::g_ctr64[sidx * block_cache::CSTRIDE64]) >> 32) != gen) {
        Block *bg = block_cache::g_prev[sidx * block_cache::RING + (gen % block_cache::RING)];
        if (bg != nullptr)
          free_offset(table->get_global_block_offset(bg) * 4096 + count);
        continue;
      }
      if (count == 4095) {
        swap_slot_static(tree_id, sidx, slot, gen);
        o_cidx = -1;  // slot just swapped -> don't cache it
      } else {
        o_cidx = sidx;
      }
      o_cbase = sbase;
      o_cgen = gen;
      return (void *)(sbase + (uint64_t)count * alloc_size);
    }
    return nullptr;  // genuine exhaustion
  }
#endif

  //v2 of malloc - handle tree_id externally.
  __device__ void * malloc(uint64_t size){

    //updated version for register sharing
    // uint alloc_count = 1;

    // // 0 = slice, 1 = block, 2 = segment
    // int alloc_level = 0;

    // if (size < smallest) size = smallest;

    uint16_t tree_id = get_tree_id_from_size(size);

    // Static-managed single-slice trees are served ENTIRELY by the fast path (it
    // owns block replacement), so the cooperative path never co-manages these
    // slots. Big allocs (tree_id >= num_trees) and any unmanaged tree (g_nblk==0)
    // fall through to the cooperative path below. GALLATIN_STATIC_COUNTER (off-block
    // one-atomic) takes precedence over the block-pointer cache when both are set.
#if defined(GALLATIN_STATIC_COUNTER)
    if (tree_id < num_trees && block_cache::g_nblk[tree_id] > 0) {
      return malloc_static(tree_id);
    }
#elif defined(GALLATIN_BLOCK_CACHE)
    if (tree_id < num_trees && block_cache::g_nblk[tree_id] > 0) {
      return malloc_cached(tree_id);
    }
#endif

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
  GALLATIN_COLD_ATTR
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
  GALLATIN_COLD_ATTR
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
  GALLATIN_COLD_ATTR
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
  GALLATIN_COLD_ATTR
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