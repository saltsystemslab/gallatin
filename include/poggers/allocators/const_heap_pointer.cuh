#ifndef CONST_HEAP_POINTER
#define CONST_HEAP_POINTER


#include <cuda.h>
#include <cuda_runtime_api.h>

#include "stdio.h"
#include "assert.h"


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif



//TODO:
//rewrite with 16 byte padding

// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

#define MASK (1ULL << 16)-1


//a pointer list managing a set section o fdevice memory

namespace poggers {


namespace allocators { 


struct internal_bitfield {
	uint counter : 5;
	uint offset : 27; 

};

union unioned_bitfield {

	internal_bitfield as_bitfield;
	uint as_uint;

};

template <typename T>
__global__ void init_block_helper_kernel(void * ext_memory_ptr){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid != 0) return;

	//printf("Starting allocation\n");

	T::init_superblock(ext_memory_ptr);
	return;

}

template <size_t allocation_size, size_t pointers_per_superblock>
struct const_heap_pointer {

	using my_type = const_heap_pointer<allocation_size, pointers_per_superblock>;

	static_assert(sizeof(internal_bitfield) == 4);

	public:

		//assert that we can wrap around to the next item
		//we need to move up to x spots in the list

		static_assert(allocation_size >= 4);

		static_assert(pointers_per_superblock <= (1ULL << (15)));

		//distance to next allocation is measured in allocation_size


		unioned_bitfield distance_and_counter;



		//uint16_t distance_to_next_next_allocation;

		__host__ __device__ static inline uint get_offset_from_mixed(unioned_bitfield ext_dist_and_counter){

			return ext_dist_and_counter.as_bitfield.offset;
		}

		__device__ static inline uint get_counter_from_mixed(unioned_bitfield ext_dist_and_counter){

			return ext_dist_and_counter.as_bitfield.counter; 
		}

		__device__ static unioned_bitfield merge_counter_and_pointer(uint counter, uint offset){

			unioned_bitfield bitfield_to_return;
			bitfield_to_return.as_bitfield.counter = counter;
			bitfield_to_return.as_bitfield.offset = offset;
			return bitfield_to_return;
		}

		__host__ __device__ const_heap_pointer(){};

		//request for a thread to allocate the entirety of a superblock to this sizing
		//necessary for HOARD implementation
		//might come back later with a parallel implementation for coop groups
		__device__ static my_type * init_superblock(void * superblock_space){



			my_type * modified_ptr = (my_type *) superblock_space;

			for (uint64_t i =0; i < pointers_per_superblock; i++){

				//printf("Initializing i: %llu\n", i);

				modified_ptr[i].distance_and_counter = my_type::merge_counter_and_pointer(0, 1);

				#if DEBUG_ASSERTS

				assert(my_type::get_offset_from_mixed(modified_ptr[i].distance_and_counter) == 1);
				assert(my_type::get_counter_from_mixed(modified_ptr[i].distance_and_counter) == 0);

				#endif

			}

			//set last pointer
			//printf("Fixing %llu\n", pointers_per_superblock-1);
			modified_ptr[pointers_per_superblock -1].distance_and_counter = my_type::merge_counter_and_pointer(0, 0);

			assert(my_type::get_offset_from_mixed(modified_ptr[pointers_per_superblock -1].distance_and_counter) == 0);
			assert(my_type::get_counter_from_mixed(modified_ptr[pointers_per_superblock -1].distance_and_counter) == 0);

			__threadfence();

			//this is the head of the list
			//printf("Ending allocation\n");

			return modified_ptr;

		}

		__device__ unioned_bitfield atomicload(){
			
			unioned_bitfield ret_bitfield;
			ret_bitfield.as_uint = (atomicCAS((unsigned int *)&distance_and_counter, 0U, 0U));
			return ret_bitfield;
		}

		__device__ my_type * next(uint offset){

			//uint16_t offset = atomicload();

			if (offset > pointers_per_superblock){
				int step_back = pointers_per_superblock - offset;

				#if DEBUG_ASSERTS

				assert(offset < 2*pointers_per_superblock);
				assert((-step_back) < pointers_per_superblock);

				#endif

				return this + step_back;

			} else if (offset == 0){

				return nullptr;

			} else {

				return this + offset;

			}

		}

		__device__ my_type * non_atomic_get_next(){

			uint offset = my_type::get_offset_from_mixed(distance_and_counter);

			// printf("distance_and_counter: %u\n", distance_and_counter);
			// printf("Offset found in non atomic: %hu\n", offset);

			return next(offset);

		}

		__device__ inline uint get_offset_from_next(my_type * next){

			if (next == nullptr){
				return 0;
			} 

			int shift = next - this;
			
			if (shift > 0){

				//next is ahead, use regular logic
				return (uint) shift;

			} else {

				uint ret_val = pointers_per_superblock - shift;

				#if DEBUG_ASSERTS

				assert(shift != 0);

				assert(ret_val > pointers_per_superblock);
				#endif

				return ret_val;

			}

		}


		//atomic to update the state based on what we saw the last time we did an atomic read
		__device__ inline bool atomic_set_value(unioned_bitfield offset, unioned_bitfield replacement){

			if (atomicCAS((unsigned int *)&distance_and_counter, (unsigned int) offset.as_uint, (unsigned int) replacement.as_uint) == (unsigned int) offset.as_uint){
				//fence for good measure
				__threadfence();
				return true;
			}

			return false;


		}

		__device__ inline bool set_next(unioned_bitfield old_counter, uint new_counter, my_type * other){

			uint replacement_next = get_offset_from_next(other);

			unioned_bitfield merged = my_type::merge_counter_and_pointer(new_counter, replacement_next);


			return atomic_set_value(old_counter, merged);

		}


		__device__ inline void * malloc(){

			//printf("Starting Malloc\n");

			//loop until we see no nodes available or we grab a node
			while (true){

				unioned_bitfield merged = atomicload();

				uint offset = my_type::get_offset_from_mixed(merged);


				
				//new operation will increment this by 1
				//we only care about exact equality, not > or <
				uint new_counter = my_type::get_counter_from_mixed(merged) + 1;


				my_type * next_node = next(offset);

				if (next_node == nullptr){
					return nullptr;
				}

				uint next_offset = my_type::get_offset_from_mixed(next_node->atomicload());

				//static grab done
				//no reads from invalid memory possible, if state changed at this point
				//its ok

				my_type * next_next_node = next_node->next(next_offset);

				//calculate where we need to point to swap out node
				//uint16_t replacement_value = get_offset_from_next(next_next_node);

				if (set_next(merged, new_counter, next_next_node)){

					//printf("Malloced succeeded\n");
					//new node is disentangled from the list!
					//convert to generic pointer and return
					return (void *) next_node;

				}

				//printf("Malloc failed\n");

			}


		}

		__device__ inline void free(void * allocation){

			my_type * casted_allocation = (my_type *) allocation;

			//free can't fail - we just loop until we make it
			while (true){



				//to swap node we need to set next on allocation
				//and then atomically set next on current
				unioned_bitfield head_merged = atomicload();

				uint head_offset = my_type::get_offset_from_mixed(head_merged);

				//new operation will increment this by 1
				//we only care about exact equality, not > or <
				uint head_new_counter = my_type::get_counter_from_mixed(head_merged) + 1;


				my_type * head_next = next(head_offset);

				//now that we 

				unioned_bitfield casted_merged = casted_allocation->atomicload();

				uint casted_new_counter = my_type::get_counter_from_mixed(casted_merged)+1;

				if (casted_allocation->set_next(casted_merged, casted_new_counter, head_next)){

					if (set_next(head_merged, head_new_counter, casted_allocation)){
						return;
					}

				}


			}

		}



		__device__ int count_heap_valid(){

			//just iterate_through heap

			//printf("Starting heap count\n");

			int counter = 0;
			my_type * heap_ptr = this;

			while (heap_ptr != nullptr){

				counter+=1;

				assert(counter <= pointers_per_superblock);

				//printf("Offset from 0: %llu\n", heap_ptr - this);

				heap_ptr = heap_ptr->non_atomic_get_next();

			}

			return counter;

		}


		__host__ static my_type * host_init_superblock(void * ext_memory_ptr){


			init_block_helper_kernel<my_type><<<1,1>>>(ext_memory_ptr);
			cudaDeviceSynchronize();

			return (my_type *) ext_memory_ptr;


		}


	

};

template <size_t bytes_per_item, size_t bytes_given>
struct bytes_given_wrapper {
   	
    using heap_ptr = const_heap_pointer<bytes_per_item, bytes_given/bytes_per_item>;
};


//manager for pointer - this is like a header / extras

//enforce even power of two


template< size_t bytes_per_item, size_t bytes_given>
struct manager {

	static_assert(bytes_given != 0 && (bytes_given & (bytes_given-1)) == 0);

	using my_type = manager<bytes_per_item, bytes_given>;

	using pointer_type = typename bytes_given_wrapper<bytes_per_item, bytes_given - sizeof(my_type)>::heap_ptr;


	//components necessary
	//pointer to next type
	//current counter of block?
	uint16_t counter;
	uint16_t work_state;
	my_type * next_node;
	//int allocated_counter;

	__device__ inline pointer_type * get_stack(){

		return (pointer_type *) (this+1);

	}

	//start up this block - init the lower block and 
	__device__ static my_type * init_heap(void * ext_pointer){

		my_type * my_heap = (my_type * ) ext_pointer;

		my_heap->counter = 0;
		my_heap->work_state = 0;
		my_heap->next_node = nullptr;

		//its ititialized after this
		pointer_type::init_superblock(my_heap->get_stack());


	}

	__device__ void * malloc(){

		return get_stack()->malloc();
	}

	__device__ void free(void * my_allocation){

		get_stack()->free(my_allocation);
	}


};


//preconditioned that alignment is at least size_in_bytes
template<size_t size_in_bytes>
void * find_manager(void * pointer){

	uint64_t pointer_as_uint = (uint64_t) pointer;

	uint64_t mask = size_in_bytes -1;

	//keep only upper bits
	uint64_t merged = pointer_as_uint & (~mask);

	return (void *) merged;

}




}

}


#endif //GPU_BLOCK_