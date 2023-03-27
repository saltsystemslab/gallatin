#ifndef POGGERS_BUDDY_ALLOCATOR
#define POGGERS_BUDDY_ALLOCATOR


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <stdio.h>
#include "assert.h"

//a pointer list managing a set section of device memory


//include reporter for generating usage reports
#include <poggers/allocators/reporter.cuh>

// #define DEBUG_ASSERTS 1

#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//define CUTOFF_SIZE 1024
#define CUTOFF_SIZE 50

#if COUNTING_CYCLES
#include <poggers/allocators/cycle_counting.cuh>
#endif



//Attachment Policy

//Insertion is always at the head
//A -> C -> D
//A -> C -> D
// B ->
// Leads to
// A -> B -> C -> D
// B sets to C, threadfence
// CAS on A->next
// after this occurs C.prev is still A
// so cas needs to take EXT value for consistency - we need to use c.next on remove instead of swapping away whatever is in A.next


//Four cases

//1) Insert and Delete


//2) Insert and Insert

//3) Delete and Insert

//4) Delete and Delete


//Can I be smarter about this?
//using single linked list?
//or maybe some sort of virtual structure on top of the list?



//struct that manages the buddy allocator?
//each item has a uint64_t next - one bit for valid - and one bit for deleted
// only items which have been properly marked as deleted can be resinserted into the list

//can clean up into 8 bytes per bit in the main array?
//the memory pointer is by default 

//we can update the valid flag in one step
//and clean pointers later?

//Would this work? yes?
//Pros - Find allocations in amortized constant time
//Pros - Easier to implement and novel
//Can quickly pointer swing without looking at main memory - this is a big plus to it not being in place
//Singly linked.

namespace poggers {


namespace allocators { 


struct buddy_node {
	uint64_t pointer_counter; 


	__device__ void init(){
		pointer_counter = 0ULL;
	}

	__device__ uint64_t inline generate_alloc_mask(){
		return 1ULL;
	}

	__device__ uint64_t inline generate_unset_alloc_mask(){
		return ~generate_alloc_mask();
	}

	__device__ uint64_t inline generate_delete_mask(){
		return 2ULL;
	}

	__device__ uint64_t inline generate_unset_delete_mask(){
		return ~generate_delete_mask();
	}

	//we know the buddy allocator is aligned to large blocks, so the first 4 bits are always available for a counter.
	// __device__ uint64_t get_offset(){

	// 						//00000011
	// 	const uint64_t mask = ~0x3;

	// 	uint64_t pointer_no_counter = pointer_counter & mask;

	// 	return (offset) pointer_no_counter;

	// }

	__device__ uint64_t inline get_next_counter(){

		const uint64_t mask = 0xf;

		


		//if this causes rollover it will get clipped by the mask.
		return (pointer_counter +1) & mask;

	}

	__device__ uint64_t get_value(){
		return pointer_counter;
	}

	__device__ void set_value(uint64_t pointer){

		uint64_t bit_mask = generate_alloc_mask();

		uint64_t delete_mask = generate_delete_mask();

		//pointer should be the same not matter what
		//these are just double checks that my assumptions are fair
		//i.e. every pointer is already aligned to at least 4 bytes.

		#if DEBUG_ASSERTS
		assert((pointer & ~bit_mask) == pointer);

		assert((pointer & ~delete_mask) == pointer);

		#endif

		pointer = (pointer & ~bit_mask) & ~delete_mask;


		//gather old bits
		uint64_t old_val = (pointer_counter & (bit_mask | delete_mask));

		pointer_counter = pointer | old_val;


		return;

	}

	//__device__ uint64_t get_val_with


	//we expect expected to already have a counter.
	//returns true if the pointer was updated.
	__device__ uint64_t try_attach_new(uint64_t expected, uint64_t new_pointer){

		uint64_t next_counter = get_next_counter();

		(atomicCAS((unsigned long long int *) &pointer_counter, (unsigned long long int) expected, (unsigned long long int) (new_pointer | next_counter)));


	}

	__device__ uint64_t get_my_ptr_as_uint(){
		return (uint64_t) this;
	}

	

};





//setup occurs in parallel for speed since we could potentially scale far out of GPU mem
//this has to be templated because cuda doesn't let objects have __global__ methods
//so this is called from a dummy host method that sets it up
template <typename memory_allocator>
__global__ void init_memory_pointers(memory_allocator * mem){


	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	//start by getting the pointers and memory info


	uint64_t items_at_top = mem->get_num_items_top();

	int max_p2 = mem->get_max_p2();

	int min_p2 = mem->get_min_p2();

	int num_levels = max_p2 - min_p2 + 1;

	uint64_t prev_passed = 0;

	//calculate if my TID is valid based off of these values

	for (int i = 0; i < num_levels; i++){

		if (tid < (items_at_top + prev_passed)){

			//we are a valid item!
			//double check that I am not the end of my segment
			//That occurs when tid - prev_passed = items_at_top -1


			//if we maintain these as relative offsets may be able to get away with uints? - no need to be ptrs as we can step backwards.
			mem->available_list[tid].init();

			if (tid-prev_passed != items_at_top -1){

				uint64_t ptr = mem->available_list[tid+1].get_my_ptr_as_uint();

				mem->available_list[tid].set_value(ptr);


			}

			return;

		}


		prev_passed += items_at_top;

		items_at_top *= 2;

	}


	return;


};


// struct buddy_node {

// 	buddy_internal_bitfield prev;
// 	buddy_internal_bitfield next;



// 	//precondition for assignment
// 	//always proceeds left->right
// 	//so our prev may change
// 	// but our next is always stable
// 	void remove_from_list(){

// 		buddy_node * prev_node = prev.get_pointer();

// 		//expected value of prev
// 		uint64_t prev_val = prev_node->next.get_value();

// 		//sanity check

// 		#if DEBUG_ASSERTS

// 		//prev should point to me
// 		//maybe not true? if changes under way
// 		assert(prev_node->next.get_pointer() == this);

// 		#endif

// 		uint64_t next_ptr = next.get_pointer();

// 		//so we know what our child is 
// 		//could someone set this before I get to it?

// 		assert(prev->next == this);

// 		next->stall_lock();

// 		assert(next->prev == this);

// 		prev->next = next;

// 		next->prev = prev;

// 		__threadfence();  

// 		next->unlock();

// 		prev->unlock();

// 		return;

// 	}


// 	//lock free variant can exist?
// 	//use cas to setup
// 	void add_to_list(buddy_node * new_node){

// 		stall_lock();

// 		next->stall_lock();

// 		new_node->stall_lock();

// 		buddy_node * node_3 = next;


// 		node_3->prev = new_node;

// 		next = new_node;

// 		new_node->prev = this;

// 		new_node->next = node_3;

// 		__threadfence();

// 		node_3->unlock();

// 		new_node->unlock();

// 		unlock();



// 	}



// };

//allocations range from 
template <int max_p2, int min_p2>
struct buddy_allocator {


	//in place linked lists - should be on avg two operations?
	//and tree of bit ops for speed

	uint64_t num_bytes;

	uint64_t num_items_top_level;

	buddy_node * available_list;

	void * memory;


	uint64_t * tree_mark_bitarray;

	using my_type = buddy_allocator<max_p2, min_p2>;


	__host__ buddy_allocator();

	// //given a num_bytes and an allocation, setup
	// __host__ buddy_allocator(void * ext_memory, uint64_t num_bytes){

	// 	init(ext_memory, num_bytes);
	// }

	// __host__ buddy_allocator(uint64_t num_bytes){

	// 	void * ext_memory;

	// 	cudaMalloc((void **)&ext_memory, num_bytes);

	// 	//TODO: memory safety check here

	// 	init(ext_memory, num_bytes);

	// }

	//return the number of items at the top level
	//this scaling affects how wide the tree is at every level.
	__device__ uint64_t get_num_items_top(){
		return num_items_top_level;
	}

	__device__ int get_max_p2(){
		return max_p2;
	}

	__device__ int get_min_p2(){
		return min_p2;
	}


	static __host__ my_type * generate_on_device(uint64_t num_bytes){

		void * ext_mem;

		cudaMalloc((void ** )&ext_mem, num_bytes);

		return generate_on_device(ext_mem, num_bytes);


	}

	//perform a device-mangaged allocation - used for big stuff.
	static __host__ my_type * generate_on_device_managed(uint64_t num_bytes){

		void * ext_mem;

		cudaMallocManaged((void ** )&ext_mem, num_bytes);

		return generate_on_device(ext_mem, num_bytes);


	}

	static __host__ my_type * generate_on_device(void * ext_memory, uint64_t num_bytes){

		//Assert alignment

		//first calculate number of pointers for each item

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		host_version->memory = ext_memory;



		uint64_t num_items_to_allocate = 0;


		uint64_t bytes_per_max_p2 = (1ULL << max_p2);

		printf("bytes per max_p2 = %llu\n", bytes_per_max_p2);

		uint64_t num_p2 = num_bytes/bytes_per_max_p2;

		printf("%llu items in the top level!\n", num_p2);

		//num_items_top_level = num_p2;

		num_items_to_allocate += num_p2;

		host_version->num_items_top_level = num_p2;

		#if DEBUG_PRINTS

		printf("Top level has %llu items\n", num_p2);

		#endif

		//Rollover safety for testing
		if (max_p2 != 0){


		for (int i = max_p2-1; i >= min_p2; i-=1){

			//each parent can split into two children
			num_p2 *= 2;

			num_items_to_allocate += num_p2;


		}

		}

		printf("Total num items: %llu\n", num_items_to_allocate);


		buddy_node * items;

		cudaMalloc((void **)& items, sizeof(buddy_node)*num_items_to_allocate);

		host_version->available_list = items;


		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		//Num items to allocate is the # of threads that should be launched
		init_memory_pointers<<<(num_items_to_allocate -1)/ 1024 + 1, 1024>>>(dev_version);

		cudaDeviceSynchronize();

		cudaFreeHost(host_version);

		return dev_version;


	}


		//release all memory associated with this allocator
		// main ptr
		// pointer table
		// alloced memory
	__host__ static void free_on_device(my_type * dev_version){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(dev_version);
		cudaFree(host_version->memory);
		cudaFree(host_version->available_list);

		cudaFreeHost(host_version);

	}








	};



}

}


#endif //GPU_BLOCK_