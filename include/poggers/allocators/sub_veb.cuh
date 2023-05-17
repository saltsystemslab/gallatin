#ifndef SUB_VEB_TREE
#define SUB_VEB_TREE
// A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy
// (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

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

// This is a chunk of a VEB tree - used by the extending VEB

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/offset_slab.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

// thank you interwebs https://leimao.github.io/blog/Proper-CUDA-Error-Checking

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

#define VEB_RESTART_CUTOFF 30

#define VEB_GLOBAL_LOAD 1
#define VEB_MAX_ATTEMPTS 15

namespace poggers {

namespace allocators {

// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

template <typename sub_veb_tree_kernel_type>
__global__ void sub_veb_report_fill_kernel(sub_veb_tree_kernel_type *tree,
                                           uint64_t num_threads,
                                           uint64_t *fill_count) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_threads) return;

  uint64_t my_fill = __popcll(tree->layers[tree->num_layers - 1].bits[tid]);

  atomicAdd((unsigned long long int *)fill_count,
            (unsigned long long int)my_fill);
}

// a layer is a bitvector used for ops
// internally, they are just uint64_t's as those are the fastest to work with

// The graders might be asking, "why the hell did you not include max and min?"
// with the power of builtin __ll commands (mostly __ffsll) we can recalculate
// those in constant time on the blocks
//  which *should* be faster than a random memory load, as even the prefetch is
//  going to be at least one cycle to launch this saves ~66% memory with no
//  overheads!
struct sub_layer {
  // make these const later
  uint64_t universe_size;
  uint64_t num_blocks;
  uint64_t *bits;
  // int * max;
  // int * min;

  __device__ void *get_mem() { return (void *)bits; }

  __device__ void reset_mem() { bits = nullptr; }

  __device__ static uint64_t static_get_num_blocks(uint64_t items_in_universe) {
    return (items_in_universe - 1) / 64 + 1;
  }

  __device__ static uint64_t static_get_size_bytes(uint64_t items_in_universe) {
    return static_get_num_blocks(items_in_universe) * sizeof(uint64_t);
  }

  __device__ uint64_t get_num_blocks() { return num_blocks; }

  __device__ uint64_t get_size_bytes() {
    return get_num_blocks() * sizeof(uint64_t);
  }

  // given a device layer initialize.
  __device__ void init_array(uint64_t *array) {
    bits = array;

    uint64_t num_blocks = get_num_blocks();

    for (uint64_t i = 0; i < num_blocks; i++) {
      bits[i] = ~0ULL;
    }

    __threadfence();

    return;
  }

  __device__ void device_init(uint64_t items_in_universe) {
    universe_size = items_in_universe;

    num_blocks = static_get_num_blocks(items_in_universe);

    bits = nullptr;
  }

  __host__ static void free_on_device(sub_layer *dev_layer) {
    sub_layer *host_layer;

    cudaMallocHost((void **)&host_layer, sizeof(sub_layer));

    cudaMemcpy(host_layer, dev_layer, sizeof(sub_layer),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(host_layer->bits);

    cudaFreeHost(host_layer);
  }

  __device__ uint64_t insert(uint64_t high, int low) {
    return atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low));
  }

  __device__ uint64_t remove(uint64_t high, int low) {
    return atomicAnd((unsigned long long int *)&bits[high], ~SET_BIT_MASK(low));
  }

  // __device__ uint64_t remove_team(uint64_t high){

  // 	cg::coalesced_group active_threads = cg::coalesced_threads();

  // 	int allocation_index_bit = 0;

  // 	uint64_t hash1 =  poggers::hashers::MurmurHash64A (&tid,
  // sizeof(uint64_t), seed);

  // }

  __device__ int inline find_next(uint64_t high, int low) {
    // printf("High is %lu, num blocks is %lu\n", high, num_blocks);
    if (bits == nullptr) {
      printf("Nullptr\n");
    }

    if (high >= universe_size) {
      printf("High issue %lu > %lu\n", high, universe_size);
      return -1;
    }

#if VEB_GLOBAL_LOAD
    poggers::utils::ldca(&bits[high]);
#endif

    if (low == -1) {
      return __ffsll(bits[high]) - 1;
    }

    return __ffsll(bits[high] & ~BITMASK(low + 1)) - 1;
  }

  // returns true if item in bitmask.
  __device__ bool query(uint64_t high, int low) {
#if VEB_GLOBAL_LOAD
    poggers::utils::ldca(&bits[high]);
#endif

    return (bits[high] & SET_BIT_MASK(low));
  }
};

struct sub_veb_tree {
  uint64_t seed;
  uint64_t total_universe;
  int num_layers;
  sub_layer *layers;
  uint64_t block_memory;

  // if all pointers point to nullptr
  // how big are we?
  // This is the number of bytes required for both the main object
  // and all layer pointers.
  static __host__ uint64_t get_size_bytes_noarray(uint64_t universe) {
    uint64_t bytes = sizeof(sub_veb_tree);

    int max_height = 64 - __builtin_clzll(universe);

    assert(max_height >= 1);
    // round up but always assume
    int ext_num_layers = (max_height - 1) / 6 + 1;

    bytes += ext_num_layers * sizeof(sub_layer);

    return bytes;
  }

  static __device__ uint64_t device_get_size_bytes_noarray(uint64_t universe) {
    uint64_t bytes = sizeof(sub_veb_tree);

    int max_height = 64 - __clzll(universe);

    assert(max_height >= 1);
    // round up but always assume
    int ext_num_layers = (max_height - 1) / 6 + 1;

    bytes += ext_num_layers * sizeof(sub_layer);

    return bytes;
  }

  __device__ uint64_t get_size_arrays() {
    uint64_t bytes = 0;

    for (int i = 0; i < num_layers; i++) {
      bytes += layers[i].get_size_bytes();
    }

    return bytes;
  }

  // given memory from the external allocator, initialize this tree component.
  static __device__ sub_veb_tree *init(void *memory, uint64_t universe,
                                       uint64_t ext_seed) {
    sub_veb_tree *tree = (sub_veb_tree *)memory;

    int max_height = 64 - __clzll(universe);

    assert(max_height >= 1);
    // round up but always assume
    int ext_num_layers = (max_height - 1) / 6 + 1;

    tree->num_layers = ext_num_layers;

    tree->total_universe = universe;

    tree->seed = ext_seed;

    tree->layers = (sub_layer *)((uint64_t)memory + sizeof(sub_veb_tree));

    uint64_t ext_universe_size = universe;

    for (int i = 0; i < ext_num_layers; i++) {
      tree->layers[ext_num_layers - 1 - i].device_init(ext_universe_size);

      ext_universe_size = (ext_universe_size - 1) / 64 + 1;
    }

    return tree;
  }

  // on an already initialized sub_veb_tree, init the arrays.
  __device__ uint64_t get_num_bytes_arrays() {
    uint64_t bytes = 0;
    for (int i = 0; i < num_layers; i++) {
      bytes += layers[i].get_size_bytes();
    }

    return bytes;
  }

  // assumes arrays have been booted correctly using the requester from
  // get_num_bytes_arrays if this isn't true this will segfault.
  __device__ void init_arrays(void *memory) {
    // printf("Booting %llx with %llx\n", (uint64_t) this, (uint64_t) memory);

    for (int i = 0; i < num_layers; i++) {
      layers[i].init_array((uint64_t *)memory);

      memory = (void *)((uint64_t)memory + layers[i].get_size_bytes());
    }

    block_memory = (uint64_t)memory;
  }

  // unsets memory and returns a handler to the memory segment used for bits.
  __device__ void *free_memory() {
    void *start = layers[0].get_mem();

    for (int i = 0; i < num_layers; i++) {
      layers[i].reset_mem();
    }

    return start;
  }

  __device__ bool float_up(int &layer, uint64_t &high, int &low) {
    layer -= 1;

    low = high & BITMASK(6);
    high = high >> 6;

    return (layer >= 0);
  }

  __device__ bool float_down(int &layer, uint64_t &high, int &low) {
    layer += 1;
    high = (high << 6) + low;
    low = -1;

    return (layer < num_layers);
  }

  // base setup - only works with lowest level
  __device__ bool remove(uint64_t delete_val) {
    uint64_t high = delete_val >> 6;

    int low = delete_val & BITMASK(6);

    int layer = num_layers - 1;

    uint64_t old = layers[layer].remove(high, low);

    if (!(old & SET_BIT_MASK(low))) return false;

    while (__popcll(old) == 1 && float_up(layer, high, low)) {
      old = layers[layer].remove(high, low);
    }

    return true;

    // assert (high == delete_val/64);
  }

  __device__ bool insert(uint64_t insert_val) {
    uint64_t high = insert_val >> 6;

    int low = insert_val & BITMASK(6);

    int layer = num_layers - 1;

    uint64_t old = layers[layer].insert(high, low);

    if ((old & SET_BIT_MASK(low))) return false;

    while (__popcll(old) == VEB_RESTART_CUTOFF && float_up(layer, high, low)) {
      old = layers[layer].insert(high, low);
    }

    return true;
  }

  // non atomic
  __device__ bool query(uint64_t query_val) {
    uint64_t high = query_val >> 6;
    int low = query_val & BITMASK(6);

    return layers[num_layers - 1].query(high, low);
  }

  __device__ __host__ static uint64_t fail() { return ~0ULL; }

  // finds the next one
  // this does one float up/ float down attempt
  // which gathers ~80% of items from testing.
  __device__ uint64_t successor(uint64_t query_val) {
    // debugging
    // this doesn't trigger so not the cause.

    uint64_t high = query_val >> 6;
    int low = query_val & BITMASK(6);

    int layer = num_layers - 1;

    while (true) {
      int found_idx = layers[layer].find_next(high, low);

      if (found_idx == -1) {
        if (layer == 0) return sub_veb_tree::fail();

        float_up(layer, high, low);
        continue;

      } else {
        break;
      }
    }

    while (layer != (num_layers - 1)) {
      low = layers[layer].find_next(high, low);

      if (low == -1) {
        return sub_veb_tree::fail();
      }
      float_down(layer, high, low);
    }

    low = layers[layer].find_next(high, low);

    if (low == -1) return sub_veb_tree::fail();

    return (high << 6) + low;
  }

  __device__ uint64_t lock_offset(uint64_t start) {
    // temporarily clipped for debugging
    if (query(start) && remove(start)) {
      return start;
    }

    // return sub_veb_tree::fail();

    while (true) {
      start = successor(start);

      if (start == sub_veb_tree::fail()) return start;

      // this successor search is returning massive values - why?
      if (remove(start)) {
        return start;
      }
    }
  }

  __device__ uint64_t malloc() {
    // make several attempts at malloc?

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t hash1 =
        poggers::hashers::MurmurHash64A(&tid, sizeof(uint64_t), seed);

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t hash2 =
        poggers::hashers::MurmurHash64A(&tid, sizeof(uint64_t), hash1);

    int attempts = 0;

    while (attempts < VEB_MAX_ATTEMPTS) {
      uint64_t index_to_start =
          (hash1 + attempts * hash2) % (total_universe - 64);

      if (index_to_start == ~0ULL) {
        index_to_start = 0;
        printf("U issue\n");
      }

      uint64_t offset = lock_offset(index_to_start);

      if (offset != sub_veb_tree::fail()) return offset;

      attempts++;
    }

    return lock_offset(0);
  }

  // take a local offset, return the block.
  __device__ offset_alloc_bitarr *get_block_from_offset(uint64_t local_offset) {
    return (offset_alloc_bitarr *)(block_memory +
                                   local_offset * sizeof(offset_alloc_bitarr));
  }

  __device__ uint64_t get_largest_allocation() { return total_universe; }

  __host__ uint64_t host_get_universe() {
    sub_veb_tree *host_version;

    cudaMallocHost((void **)&host_version, sizeof(sub_veb_tree));

    cudaMemcpy(host_version, this, sizeof(sub_veb_tree),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    uint64_t ret_value = host_version->total_universe;

    cudaFreeHost(host_version);

    return ret_value;
  }

  __host__ uint64_t report_fill() {
    uint64_t *fill_count;

    cudaMallocManaged((void **)&fill_count, sizeof(uint64_t));

    cudaDeviceSynchronize();

    fill_count[0] = 0;

    cudaDeviceSynchronize();

    uint64_t max_value = report_max();

    uint64_t num_threads = max_value / 64;

    sub_veb_report_fill_kernel<sub_veb_tree>
        <<<(num_threads - 1) / 512 + 1, 512>>>(this, num_threads, fill_count);

    cudaDeviceSynchronize();

    uint64_t return_val = fill_count[0];

    cudaFree(fill_count);

    return return_val;
  }

  __host__ uint64_t report_max() {
    // return 1;
    return host_get_universe();
  }

  // //teams work togther to find new allocations
  // __device__ uint64_t team_malloc(){

  // 	cg::coalesced_group active_threads = cg::coalesced_threads();

  // 	int allocation_index_bit = 0;

  // }
};

}  // namespace allocators

}  // namespace poggers

#endif  // End of VEB guard