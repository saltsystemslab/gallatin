#ifndef STACK_OF_STACKS
#define STACK_OF_STACKS


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <poggers/allocators/aligned_stack.cuh>

#include "stdio.h"
#include "assert.h"


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

#define MASK (1ULL << 16)-1


//a pointer list managing a set section o fdevice memory

namespace poggers {


namespace allocators { 


template <size_t stacks_per_manager, size_t bytes_per_stack, bool auto_dealloc>
struct stack_of_stacks_manager{

	static_assert(bytes_per_stack != 0 && (bytes_per_stack & (bytes_per_stack-1)) == 0);
	//using offset_type = first_matching_address<bytes_per_item>;

	//using bytes_remaining_type = find_offset<bytes_per_stack, offset_type::value>;



	//the internals of this are basically a giant aligned_heap_ptr that serves
	// stack-size allocations
	using pointer_type = aligned_heap_ptr<bytes_per_stack*stacks_per_manager>;

	using stack_type = aligned_manager<bytes_per_stack, auto_dealloc>;

	using my_type = stack_of_stacks_manager<stacks_per_manager, bytes_per_stack, auto_dealloc>;





	//unioned_bitfield stack_head;

	pointer_type stack_head;

	uint counter;

	my_type * ptr_to_next;
	my_type * ptr_to_prev;
	uint64_t spacing;


	//funcs needed
	//init stack
	//destruct stack
	//malloc
	//free


	//some useful helpers
	//malloc next
	//lock structure

	__device__ static my_type * init_stack_of_stacks(void * ext_address){

		#if DEBUG_PRINTS
		printf("Initializing new stack of stacks with %llu bytes per item, %llu overall\n", bytes_per_stack, bytes_per_stack*stacks_per_manager);
		#endif

		my_type * new_stack = (my_type * ) ext_address;

		//new_stack->stack_head.as_uint = 0;

		new_stack->counter = 0;

		new_stack->ptr_to_next = nullptr; 
		new_stack->ptr_to_prev = nullptr;

		//uint64_t address_as_uint = (uint64_t) ext_address;

		//void * startup_address = (void *) (address_as_uint + offset_type::value);

		//return the pointer to the first valid
		pointer_type * head = pointer_type::init_stack(ext_address, (unsigned long) sizeof(my_type), bytes_per_item);

		//init_stack(void * superblock_space, int bytes_in_use, int bytes_per_item){

		//pointer_type::init_superblock(startup_address);

		new_stack->stack_head.set_next_atomic(head);

		return new_stack;


	}

	__device__ static my_type * init_stack_of_stacks_free_list(header * free_list, int bytes_per_item){

		//request an address such that all stacks that are a part of me are aligned
		void * address_to_use = free_list->malloc_aligned(bytes_per_stack*stacks_per_manager+sizeof(my_type), bytes_per_stack, -1*sizeof(my_type));

		if (address_to_use == nullptr) return nullptr;

		return my_type::init_stack_of_stacks(address_to_use, bytes_per_item);

	}


	__device__ static void free_stack_of_stacks(header * free_list, my_type * stack){

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



	__device__ void * malloc_stack(){

		return stack_head.malloc();

	}

	__device__ void free_stack(void * ext_address){

		//pointer_type * ext_as_ptr = (void *) ext_address;

		stack_head.free(ext_address);

	}

	//this doesn't exist?
	// __device__ static void static_free(void * ext_address){

	// 	pointer_type * ext_as_obj = (pointer_type *) ext_address;



	// 	my_type * home_as_stack = (my_type *) ext_as_obj->get_home_address();

	// 	home_as_stack->free(ext_address);

	// }

	// __device__ static uint64_t get_home_address_uint(void * ext_address){

	// 	pointer_type * ext_as_obj = (pointer_type * ) ext_address;

	// 	return ext_as_obj->get_home_address();

	// }

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


};





}

}


#endif //GPU_BLOCK_