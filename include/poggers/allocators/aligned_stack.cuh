#ifndef ALIGNED_HEAP_STACK
#define ALIGNED_HEAP_STACK


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include "stdio.h"
#include "assert.h"


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#if COUNTING_CYCLES
#include <poggers/allocators/cycle_counting.cuh>
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

template <typename stack>
__global__ void stack_init_kernel(uint64_t superblock_as_uint, int bytes_in_use, int bytes_per_item){


	stack::init_stack_helper(superblock_as_uint, bytes_in_use, bytes_per_item);
}


template <size_t bytes_used>
struct aligned_heap_ptr {

	using my_type = aligned_heap_ptr<bytes_used>;

	static_assert(sizeof(internal_bitfield) == 4);

	public:

		//assert that we can wrap around to the next item
		//we need to move up to x spots in the list

		static_assert(bytes_used >= 4);

		//assert that this heap ptr is aligned to a power of two
		//this is important for get_home()
		static_assert((bytes_used & (bytes_used -1)) == 0);
		static_assert(bytes_used != 0);

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

		__host__ __device__ aligned_heap_ptr(){};


		__device__ static void init_stack_helper(uint64_t superblock_as_uint, int bytes_in_use, int bytes_per_item){

			uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

			my_type * modified_ptr = (my_type *) (superblock_as_uint + bytes_in_use + tid*bytes_per_item);

			my_type * next_modified_ptr = (my_type *) (superblock_as_uint + bytes_in_use + (tid+1)*bytes_per_item);

			if (modified_ptr->get_home_address() != superblock_as_uint){
				return;
			}

			if (next_modified_ptr->get_home_address() == superblock_as_uint){
				uint64_t next_offset = modified_ptr->get_offset_from_next(next_modified_ptr);
				modified_ptr->distance_and_counter = my_type::merge_counter_and_pointer(0, next_offset);

			} else {
				modified_ptr->distance_and_counter = my_type::merge_counter_and_pointer(0, 0);

			}
			
			return;






		}

		__device__ static my_type * init_stack_team(void * superblock_space , int bytes_in_use, int bytes_per_item){

			#if COUNTING_CYCLES
				
			uint64_t stack_init_counter_start = clock64();

			#endif

			uint64_t superblock_as_uint = (uint64_t) superblock_space;

			my_type * orig_modified_ptr = (my_type *) (superblock_as_uint + bytes_in_use);

			uint64_t max_calls = (bytes_used-1)/bytes_per_item+1;

			stack_init_kernel<my_type><<<(max_calls -1)/512+1, 512>>>(superblock_as_uint, bytes_in_use, bytes_per_item);

			cudaDeviceSynchronize();

			__threadfence();

			#if COUNTING_CYCLES
				
				uint64_t stack_init_counter_end = clock64();

				uint64_t aligned_stack_init_time = (stack_init_counter_end - stack_init_counter_start)/COMPRESS_VALUE;

				atomicAdd((unsigned long long int *) &stack_init_counter, (unsigned long long int) aligned_stack_init_time);

				atomicAdd((unsigned long long int *) &stack_init_traversals, (unsigned long long int) 1);


			#endif


			return orig_modified_ptr;




		}

		__device__ static my_type * init_stack(void * superblock_space, int bytes_in_use, int bytes_per_item){

			//printf("Booting stack\n");

			my_type * my_stack = init_stack_team(superblock_space, bytes_in_use, bytes_per_item);

			//my_type * my_stack = init_stack_old(superblock_space, bytes_in_use, bytes_per_item);

			//printf("Counted val: %d\n", my_stack->count_heap_valid());

			return my_stack;

		}

		//request for a thread to allocate the entirety of a superblock to this sizing
		//necessary for HOARD implementation
		//might come back later with a parallel implementation for coop groups
		__device__ static my_type * init_stack_old(void * superblock_space, int bytes_in_use, int bytes_per_item){

			#if COUNTING_CYCLES
				
			uint64_t stack_init_counter_start = clock64();

			#endif



			uint64_t superblock_as_uint = (uint64_t) superblock_space;
			uint64_t i = 0;

			my_type * modified_ptr = (my_type *) (superblock_as_uint + bytes_in_use);

			my_type * orig_modified_ptr = modified_ptr;



			//run until part of this block
			while (true){


				my_type * next_modified_ptr = (my_type *) (superblock_as_uint + bytes_in_use + (i+1)*bytes_per_item);

				uint next_offset = modified_ptr->get_offset_from_next(next_modified_ptr);

				modified_ptr->distance_and_counter = my_type::merge_counter_and_pointer(0, next_offset);


				i+=1;

				if (next_modified_ptr->get_home_address() != superblock_as_uint){
					break;
				}

				modified_ptr = next_modified_ptr;
				

			}

			//modified_ptr points to the last item;
			modified_ptr->distance_and_counter = my_type::merge_counter_and_pointer(0,0);

			//set last pointer
			//printf("Fixing %llu\n", pointers_per_superblock-1);
			//modified_ptr[pointers_per_superblock -1].distance_and_counter = my_type::merge_counter_and_pointer(0, 0);

			//assert(my_type::get_offset_from_mixed(modified_ptr[pointers_per_superblock -1].distance_and_counter) == 0);
			//assert(my_type::get_counter_from_mixed(modified_ptr[pointers_per_superblock -1].distance_and_counter) == 0);

			__threadfence();

			#if COUNTING_CYCLES
				
				uint64_t stack_init_counter_end = clock64();

				uint64_t aligned_stack_init_time = (stack_init_counter_end - stack_init_counter_start)/COMPRESS_VALUE;

				atomicAdd((unsigned long long int *) &stack_init_counter, (unsigned long long int) aligned_stack_init_time);

				atomicAdd((unsigned long long int *) &stack_init_traversals, (unsigned long long int) 1);


			#endif

			//this is the head of the list
			//printf("Ending allocation\n");

			return orig_modified_ptr;

		}

		__device__ unioned_bitfield atomicload(){
			
			unioned_bitfield ret_bitfield;
			ret_bitfield.as_uint = (atomicCAS((unsigned int *)&distance_and_counter, 0U, 0U));
			return ret_bitfield;
		}

		__device__ my_type * next(uint offset){

			//uint16_t offset = atomicload();


			//gets the address to the next key based on offset

			if (offset == 0){
				return nullptr;
			}

			uint offset_shifted = offset << 2;

			uint64_t home = get_home_address();

			home += offset_shifted;

			return (my_type *) home;

		}

		__device__ my_type * non_atomic_get_next(){

			uint offset = my_type::get_offset_from_mixed(distance_and_counter);

			// printf("distance_and_counter: %u\n", distance_and_counter);
			// printf("Offset found in non atomic: %hu\n", offset);

			return next(offset);

		}

		__device__ uint64_t get_home_address(){

			uint64_t this_as_uint = (uint64_t) this;

			uint64_t mask = bytes_used - 1;

			this_as_uint = this_as_uint & (~mask);

			return this_as_uint;

		}

		__device__ inline uint get_offset_from_next(my_type * next){

			if (next == nullptr){
				return 0;
			} 

			uint64_t next_as_uint = (uint64_t) next;

			uint64_t shift = next_as_uint - get_home_address();

			//int shift = next - this;
			
			//shifts always record themselves as shift/4, as the minimum size allowed is 4 bytes (anyone else is secretly promoted!)
			//this allows us to store 4x more items per block then before while still

			return shift >> 2;

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

			uint replacement_next = my_type::get_offset_from_next(other);

			unioned_bitfield merged = my_type::merge_counter_and_pointer(new_counter, replacement_next);


			return atomic_set_value(old_counter, merged);

		}

		__device__ inline bool set_next_atomic(my_type * next_node){

		
			unioned_bitfield merged = atomicload();

			//uint old_offset = my_type::get_offset_from_mixed(merged);

			uint new_counter = my_type::get_offset_from_mixed(merged)+1;

			//uint next_offset = my_type::get_offset_from_mixed(next_node);



			return set_next(merged, new_counter, next_node);



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

			//1 less than expected, head is not available for malloc.
			return counter-1;

		}


	

};

// template <size_t bytes_per_item, size_t bytes_given>
// struct bytes_given_wrapper {
   	
//     using heap_ptr = aligned_heap_ptr<bytes_per_item, bytes_given/bytes_per_item>;
// };






// //template metaprogramming to find the next side
// template <std::size_t bytes_per_item>
// struct first_matching_address{
// 	static const uint64_t value = std::max(bytes_per_item, (std::size_t) 16);
// };

// template <std::size_t bytes_given, std::size_t offset>
// struct find_offset{

// 	static const uint64_t remaining = bytes_given-offset;

// };

//pieces used by manager
//one uint for flags / offset - this is the rough ptr
// one for a fill counter and "locking" flags
// on uint64_t for the ptr to next



template <size_t bytes_given, bool auto_dealloc>
struct aligned_manager{

	static_assert(bytes_given != 0 && (bytes_given & (bytes_given-1)) == 0);

	//using offset_type = first_matching_address<bytes_per_item>;

	//using bytes_remaining_type = find_offset<bytes_given, offset_type::value>;

	//using pointer_type = typename bytes_given_wrapper<bytes_given>::heap_ptr;
	using pointer_type = aligned_heap_ptr<bytes_given>;

	using my_type = aligned_manager<bytes_given, auto_dealloc>;



	//unioned_bitfield stack_head;

	pointer_type stack_head;

	uint counter;

	my_type * ptr_to_next;
	my_type * ptr_to_prev;

	void * my_suballocator;


	//funcs needed
	//init stack
	//destruct stack
	//malloc
	//free


	//some useful helpers
	//malloc next
	//lock structure

	__device__ static my_type * init_stack(void * ext_address, int bytes_per_item){

		#if DEBUG_PRINTS
		printf("Initializing new stack with %d bytes per item\n", bytes_per_item);
		#endif

		my_type * new_stack = (my_type * ) ext_address;

		//new_stack->stack_head.as_uint = 0;

		new_stack->counter = 0;

		new_stack->ptr_to_next = nullptr; 
		new_stack->ptr_to_prev = nullptr;

		//uint64_t address_as_uint = (uint64_t) ext_address;

		//void * startup_address = (void *) (address_as_uint + offset_type::value);

		//return the pointer to the first valid
		pointer_type * head = pointer_type::init_stack(ext_address, max( (unsigned long) bytes_per_item, (unsigned long) sizeof(my_type)), bytes_per_item);

		//init_stack(void * superblock_space, int bytes_in_use, int bytes_per_item){

		//pointer_type::init_superblock(startup_address);

		new_stack->stack_head.set_next_atomic(head);

		return new_stack;


	}

	__device__ static my_type * init_from_free_list(header * free_list, int bytes_per_item){

		void * address_to_use = free_list->malloc_aligned(bytes_given, bytes_given, 0);

		if (address_to_use == nullptr) return nullptr;

		return my_type::init_stack(address_to_use, bytes_per_item);

	}


	__device__ static void free_stack(header * free_list, my_type * stack){

		if constexpr(auto_dealloc){

			assert (counter.as_bitfield.offset == 0);


		}

		free_list->free(stack);



	}

	__device__ unioned_bitfield atomicload_stack_head(){
			
			unioned_bitfield ret_bitfield;
			ret_bitfield.as_uint = (atomicCAS((unsigned int *)&stack_head, 0U, 0U));
			return ret_bitfield;
	}

	__device__ unioned_bitfield atomicload_counter(){
			
			unioned_bitfield ret_bitfield;
			ret_bitfield.as_uint = (atomicCAS((unsigned int *)&counter, 0U, 0U));
			return ret_bitfield;
	}



	__device__ void * malloc(){

		#if COUNTING_CYCLES
			
		uint64_t stack_counter_start = clock64();

		#endif

		void * my_malloc = stack_head.malloc();

		#if COUNTING_CYCLES
			
		uint64_t stack_counter_end = clock64();

		uint64_t aligned_stack_total_time = (stack_counter_end - stack_counter_start)/COMPRESS_VALUE;

		atomicAdd((unsigned long long int *) &stack_counter, (unsigned long long int) aligned_stack_total_time);

		atomicAdd((unsigned long long int *) &stack_traversals, (unsigned long long int) 1);


		#endif

		return my_malloc;

	}

	__device__ void free(void * ext_address){

		//pointer_type * ext_as_ptr = (void *) ext_address;

		stack_head.free(ext_address);

	}

	__device__ static void static_free(void * ext_address){

		pointer_type * ext_as_obj = (pointer_type *) ext_address;



		my_type * home_as_stack = (my_type *) ext_as_obj->get_home_address();

		home_as_stack->free(ext_address);

	}

	__device__ static uint64_t get_home_address_uint(void * ext_address){

		pointer_type * ext_as_obj = (pointer_type * ) ext_address;

		return ext_as_obj->get_home_address();

	}

	__device__ my_type * get_next(){
		return ptr_to_next;
	}


	__device__ my_type * get_next_atomic(){

		return (my_type *) atomicCAS((unsigned long long int *)&ptr_to_next, 0ULL, 0ULL);
	}

	__device__ my_type * get_prev_atomic(){

		return (my_type *) atomicCAS((unsigned long long int *)&ptr_to_prev, 0ULL, 0ULL);

	}

	//__device__ my_type * set_next_atomic()

	__device__ void set_next(my_type * next){

		ptr_to_next = next;

		__threadfence();

	}

	__device__ void set_prev(my_type * next){
		ptr_to_prev = next;
		__threadfence();
	}

	__device__ void set_next_atomic(my_type * next){

		atomicExch((unsigned long long int *)&ptr_to_next, (unsigned long long int) next);

	}

	__device__ void set_prev_atomic(my_type * prev){

		atomicExch((unsigned long long int *)&ptr_to_prev, (unsigned long long int) prev);

	}

	//attempt to grab the lock
	//if we fail, that simply means that someone else is already on it!
	__device__ bool request_lock(){

		return poggers::helpers::typed_atomic_write<uint>(&counter, 0U, 1U);


	}


	//eventually grab the lock
	//this is useful for grabbing the head node when we know we must eventually succeed
	__device__ void stall_lock(){

		while (!request_lock());


	}

	__device__ void free_lock(){
		poggers::helpers::typed_atomic_write<uint>(&counter, 1U, 0U);
	}

	//alias cause I'm bad at remembering names.
	__device__ void unlock(){
		free_lock();
	}


	//PROFILING HELPERS 

	//count total allocations available, if you know what my size is 
	__device__ int count_total(int alignment_size){

		return (bytes_given - sizeof(my_type))/(alignment_size);

	}

	//count total allocations used
	__device__ int count_available(int alignment_size){

		return stack_head.count_heap_valid();

	}

	//raw bytes available
	__device__ int total_bytes(int alignment_size){
		return bytes_given;
	}

	//raw bytes used
	__device__ int bytes_used(int alignment_size){
		return total_bytes(alignment_size) - count_available(alignment_size)*alignment_size;
	}

	//End of helpers


};





}

}


#endif //GPU_BLOCK_