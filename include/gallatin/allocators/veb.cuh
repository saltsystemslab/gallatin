#ifndef GALLATIN_VEB_TREE
#define GALLATIN_VEB_TREE
// A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy
// (hunter@cs.utah.edu)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
// portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/veb_components.cuh>
#include <gallatin/allocators/murmurhash.cuh>



namespace gallatin {

namespace internals {

// define macros



//helper kernels

template <uint size> 
__global__ void init_component_kernel(gallatin::internals::internal_bitarray<size> * layer, uint64_t n_components, uint64_t bits_to_set){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_components) return;

  if (tid != n_components-1){
    if (!layer[tid].set_contiguous(gallatin::internals::internal_bitarray<size>::n_bits, 0)){
      #if GALLATIN_DEBUG
      write_global_log(10, tid, n_components);
      #endif
    }
  } else {

    uint remainder = bits_to_set-gallatin::internals::internal_bitarray<size>::n_bits*(n_components-1);

    if (!layer[tid].set_contiguous(remainder, 0)){
      #if GALLATIN_DEBUG
      write_global_log(10, tid, n_components);
      #endif
    }

  }

}


template <uint component_size>
struct veb {

  using my_type = veb<component_size>;
  using component_type = gallatin::internals::internal_bitarray<component_size>;


  uint32_t n_levels;

  //# of bits at the lowest level.
  uint32_t max_fanout;

  //7 levels allows for 32*1024^3 objects even for the smallest case.
  //I'm going to say that this is a reasonable restriction for this use case
  //and that avoiding 1 indirection per lookup is more ideal.
  //also makes the tree back nicely into 64 bytes.
  component_type * levels[7];


  static my_type * generate_on_device(uint32_t n_bits){

    my_type * host_version = gallatin::utils::get_host_version<my_type>();

    host_version->max_fanout = n_bits;

    uint64_t ext_n_levels = 0;

    //executes the loop one additional time to set the last layer.
    bool final_layer_set = false;

    while (n_bits > component_type::n_bits || !final_layer_set) {

      if (n_bits <= component_type::n_bits){
        final_layer_set = true;
      }

      uint64_t n_components = (n_bits-1)/component_type::n_bits+1;

      //printf("Setting level %lu with %u bits and %u components of size %u\n", ext_n_levels, n_bits, n_components, component_type::n_bits);

      host_version->levels[ext_n_levels] = gallatin::utils::get_device_version<component_type>(n_components);

      cudaMemset(host_version->levels[ext_n_levels], 0, sizeof(component_type)*n_components);

      init_component_kernel<component_size><<<(n_components-1)/256+1, 256>>>(host_version->levels[ext_n_levels], n_components, n_bits);

      n_bits = (n_bits-1)/component_type::n_bits+1;

      ext_n_levels++;

    }

    host_version->n_levels = ext_n_levels;

    #if GALLATIN_DEBUG

    if (host_version->n_levels > VEB_MAX_LEVELS){
       printf("\033[1;31m[ ERROR    ]\033[1;0m Van Emde Boas Tree exceeded size: Max size %u exceeds limit of %u.\n", host_version->n_levels, VEB_MAX_LEVELS);
    }

    #endif

    return gallatin::utils::move_to_device_nowait<my_type>(host_version);



  }

  //generate tree with no 
  static my_type * generate_on_device_cleared(uint32_t n_bits){

    my_type * host_version = gallatin::utils::get_host_version<my_type>();

    host_version->max_fanout = n_bits;

    uint64_t ext_n_levels = 0;

    //executes the loop one additional time to set the last layer.
    bool final_layer_set = false;

    while (n_bits > component_type::n_bits || !final_layer_set) {

      if (n_bits <= component_type::n_bits){
        final_layer_set = true;
      }

      uint64_t n_components = (n_bits-1)/component_type::n_bits+1;

      //printf("Setting level %lu with %u bits and %u components of size %u\n", ext_n_levels, n_bits, n_components, component_type::n_bits);

      host_version->levels[ext_n_levels] = gallatin::utils::get_device_version<component_type>(n_components);

      cudaMemset(host_version->levels[ext_n_levels], 0, sizeof(component_type)*n_components);

      n_bits = (n_bits-1)/component_type::n_bits+1;

      ext_n_levels++;

    }

    host_version->n_levels = ext_n_levels;

    #if GALLATIN_DEBUG

    if (host_version->n_levels > VEB_MAX_LEVELS){
       printf("\033[1;31m[ ERROR    ]\033[1;0m Van Emde Boas Tree exceeded size: Max size %u exceeds limit of %u.\n", host_version->n_levels, VEB_MAX_LEVELS);
    }

    #endif

    return gallatin::utils::move_to_device_nowait<my_type>(host_version);



  }


  static void free_on_device(my_type * dev_version){

    my_type * host_version = gallatin::utils::move_to_host<my_type>(dev_version);

    for (uint i = 0; i < host_version->n_levels; i++){
      cudaFree(host_version->levels[i]);
    }

    cudaFreeHost(host_version);

  }


  __device__ bool remove(uint32_t index){

    #if GALLATIN_DEBUG
    if (index >= max_fanout){
      write_global_log(8, index, max_fanout);
    }
    #endif

    uint32_t level = 0;

    uint32_t lower_index;

    bool set=false;

    while (level < n_levels){

      uint high = index/component_type::n_bits;
      uint low = index % component_type::n_bits;


      //component_type alt_copy = levels[level][high].ld_acq();

      //alt_copy.print();

      component_type load_copy;

      if (!levels[level][high].unset_loc_atomic(low, load_copy)){

        //printf("Setting level %u, index %u, high %u, low %u\n", level, index, high, low);
        //load_copy.print();
        return set;
      }

      if (level != 0){

        if (levels[level-1][lower_index].ld_acq().popc() != 0){
          //rollback
          levels[level][high].set_loc_atomic(low, load_copy);
          return set;
        }

      }

      set = true;

      if (load_copy.popc() != 1) return set;

      lower_index = high;

      index = high;

      level++;

    }


    return set;

  }


  __device__ bool query(uint32_t index){

    #if GALLATIN_DEBUG

    if (index >= max_fanout){
      write_global_log(9, index, max_fanout);
    }

    #endif

    uint64_t high = index/component_type::n_bits;
    uint64_t low = index % component_type::n_bits;

    component_type load_copy = levels[0][high].ld_acq();

    return load_copy.query(low);

  }


  __device__ bool insert(uint32_t index){

    uint32_t level = 0;


    #if GALLATIN_DEBUG

    if (index >= max_fanout){
      write_global_log(7, index, max_fanout);
    }

    #endif


    bool set=false;

    while (level < n_levels){

      uint high = index/component_type::n_bits;
      uint low = index % component_type::n_bits;

      component_type load_copy;

      if (!levels[level][high].set_loc_atomic(low, load_copy)){
        return set;
      }

      set = true;

      if (load_copy.popc() != 0) return set;

      index = high;
      level++;

    }


    return set;

  }

  //fail state is marked by all bits at 0.
  constexpr static __device__ uint fail(){
    return ~0U;
  }


  __device__ bool out_of_bounds(uint index, uint level){


    return index >= (max_fanout-1/(component_type::n_bits*level)+1);
  }


  //find the next item
  //does not include the original index.
  __device__ uint find_first_from(uint index){


    uint level = 0;


    // uint high = index/component_type::n_bits;
    // uint low = index % component_type::n_bits;

    bool up = true;

    while (true){

      uint high = index/component_type::n_bits;
      uint low = index % component_type::n_bits;

      //should be low+1 to clip current value.
      //this allows it to always progress as prebious values are never observed.
      int first = levels[level][high].ld_acq().ffs(low+1*up)-1;

      if (first == -1){

        //move up a level or exit
        index = high;

        level++;

        up = true;

        if (level >= n_levels || out_of_bounds(index, level)){
          return my_type::fail();
        }

      } else {



        index = high*component_type::n_bits+first;

        if (level == 0){
          return index;
        }

        up = false;

        level--;

        index = index*component_type::n_bits;


      }


    }


  }

  __device__ uint find_first(uint index){

    if (query(index)) return index;

    return find_first_from(index);

  }


  __device__ uint claim_first(uint index){

    uint found_index = find_first(index);

    while (found_index != fail()){

      if (remove(found_index)){
        return found_index;
      }

      found_index = find_first(found_index);

    }

    return fail();

  }


  __device__ uint find_random(){

    uint64_t seed = gallatin::utils::get_clock_time()*gallatin::utils::get_tid();

    uint index = seed % max_fanout;

    uint back_half = find_first(index);

    if (back_half != fail()){
      return back_half;
    }

    return find_first(0);

  }





};

}  // namespace allocators

}  // namespace gallatin

#endif  // End of VEB guard