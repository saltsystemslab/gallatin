#ifndef GALLATIN_POISON
#define GALLATIN_POISON

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/global_allocator.cuh>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>



namespace cg = cooperative_groups;

//Poisoned pointers make active detections
namespace gallatin {

namespace pointers {


  //poison pin - construction of a dereference
  template <typename T>
  struct poison_pin {

    T * host_ref;


    __device__ static void set_poison(char * memory, uint64_t true_size, uint64_t alloc_size){

      //cursed cast.

      char marker = (char) 253;

      uint64_t * start_memory = (uint64_t *) memory;

      start_memory[0] = true_size;



      start_memory[1] = alloc_size;

      char * end_memory = (memory+alloc_size+16);



      for (int i =0; i < true_size-16-alloc_size; i++){

        end_memory[i] = marker;


      }

      __threadfence();



    }

    __device__ void check_poison(char * memory){

      char marker = (char) 253;

      uint64_t * start_memory = (uint64_t *) memory;

      uint64_t true_size = start_memory[0];

      uint64_t alloc_size = start_memory[1];

      char * end_memory = (memory+alloc_size+16);


      for (int i = 0; i < true_size-16-alloc_size; i++){


        if (end_memory[i] != marker){
          GALLATIN_TODO(replace global log with cudaLog when it is ready.)
          gallatin::internals::write_global_log(35, (uint64_t) (memory+16), i);
          return;
        }

      }


    }

    //check for dereference operator
    //when attempting to dereference based on this [] operator
    // we can use the alloc size to determine if this would be out of bounds.
    __device__ void check_get_OOB(std::size_t idx){

      uint64_t * start_memory = (uint64_t *) (((char *) host_ref) - 16);

      uint64_t true_size = start_memory[0];

      uint64_t alloc_size = start_memory[1];

      //alloc offset is # of bytes to start write
      //this should be solidly inside.
      uint64_t alloc_offset = sizeof(T)*idx;

      if (alloc_offset >= alloc_size){
        gallatin::allocators::write_global_log(36, (uint64_t) host_ref, idx, alloc_offset-alloc_size+1);
      }

    }

    __device__ poison_pin(char * memory){

      //needs to be moved 16 bytes as alignment might be off otherwise.
      host_ref = (T *) (memory+16);

    }

    __device__ ~poison_pin(){


      char * memory_start = ((char *) host_ref) - 16;

      check_poison(memory_start);

    }


    __device__ T* _get(){
      return host_ref;
    }

    __device__ T * operator->() {

      return _get();
    }

    __device__ T& operator[](std::size_t idx){

      check_get_OOB(idx);

      return _get()[idx];
    }


  };


  template<typename T>
  struct poison {

    char * internal_reference;

    //takes generic memory with 16 byte alignment.
    __device__ poison(char * external_reference, uint64_t memory_needed, uint64_t true_memory){
      internal_reference = external_reference;

      poison_pin<T>::set_poison(internal_reference, true_memory, memory_needed);

    }

    __device__ poison(){
      internal_reference = nullptr;
    }

    __device__ static poison<T> get_poisoned_reference(uint64_t n_copies=1){

      //get next_size_up

      uint64_t memory_needed = sizeof(T)*n_copies;

      uint64_t true_memory = 1ULL << gallatin::utils::get_first_bit_bigger(memory_needed+32);

      char * memory = (char *) gallatin::allocators::global_malloc(true_memory);

      //if this fails nullptr - dereference


      return poison<T>(memory, memory_needed, true_memory);

    }


    __device__ void free(){

      auto final_pin = _get();

      gallatin::allocators::global_free(internal_reference);
      internal_reference = nullptr;
    }


    __device__ poison_pin<T> _get(){
      return poison_pin<T>(internal_reference);
    }

    __device__ poison_pin<T> operator->() {

      return _get();

    }

    __device__ T& operator[](std::size_t idx){

      return _get()[idx];
    }



  };


}  // namespace allocators

}  // namespace gallatin

#endif  // GPU_BLOCK_