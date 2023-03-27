#ifndef FREE_LIST
#define FREE_LIST


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


#define LOCK_MASK (1ULL << 63)
#define ALLOCED_MASK (1ULL << 62)
#define ENDPOINT_MASK (1ULL << 61)


//define CUTOFF_SIZE 1024
#define CUTOFF_SIZE 50

#if COUNTING_CYCLES
#include <poggers/allocators/cycle_counting.cuh>
#endif






namespace poggers {


namespace allocators { 





struct header;

template <typename free_list_type>
__global__ void host_malloc_aligned_kernel(free_list_type * heap, void ** writable_address, uint64_t num_bytes, uint64_t alignment, int offset){


	void * allocation = heap->malloc_aligned(num_bytes, alignment, offset);

	writable_address[0] = allocation;

	return;


}

struct header_lock_dist {


	//bit-flags

	//0: locked
	//1: allocated
	//2: endpoint


	
	uint64_t size : 61;

	uint16_t lock : 3;

};

union header_bitfield {

	header_lock_dist as_bitfield;

	uint64_t as_uint;

	__device__ bool lock(){


		uint64_t old = atomicOr((unsigned long long int *) this, LOCK_MASK);

		old = old & LOCK_MASK;

		return !old;
		

	}

	__device__ void unlock(){

		atomicAnd((unsigned long long int *)this, ~LOCK_MASK);

		return;
	}


};

struct atomicLock{

	header_bitfield * header_to_lock;

	

	__device__ inline atomicLock(header * head);

	__device__ inline ~atomicLock();

};



template <typename header>
__global__ void init_heap_kernel(void * allocation, uint64_t num_bytes){


	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid != 0) return;

	header::init_heap_global(allocation, num_bytes);

}

//header alignment
//32 bytes for main structure
//16 at end for footer & even spacing

struct header {

	header_bitfield lock_and_size;
	uint64_t empty;
	header * next_ptr;
	header * prev_ptr;

	//16 byte footer included with each header!


	__device__ header_bitfield * get_lock_and_size_addr(){
		return &lock_and_size;
	}

	__device__ bool lock(){


		return lock_and_size.lock();

	}


	__device__ void stall_lock(){

		while (!lock_and_size.lock()){

			//printf("%p stalled on lock\n", this);
		}

	}

	__device__ void unlock(){


		return lock_and_size.unlock();		

	}

	__device__ uint64_t atomic_get(){

		header_bitfield * address_of_lock = &lock_and_size;

		return atomicOr((unsigned long long int *) address_of_lock, 0ULL);
	}

	__device__ void alloc(){

		atomicOr((unsigned long long int *) (&lock_and_size), ALLOCED_MASK);
		return;
	}

	__device__ void dealloc(){

		atomicAnd((unsigned long long int *)(&lock_and_size), ~ALLOCED_MASK);
		return;
	}

		__device__ bool replace(header_bitfield old, header_bitfield new_status){


		return (atomicCAS((unsigned long long int *)(&lock_and_size), old.as_uint, new_status.as_uint) == old.as_uint);

	}

	__device__ bool check_alloc(){


		return atomic_get() & ALLOCED_MASK;

	}

	__device__ bool check_lock(){
		return atomic_get() & LOCK_MASK;
	}

	__device__ bool check_endpoint(){

		return atomic_get() & ENDPOINT_MASK;

	}

	__device__ uint64_t * get_footer(){

		char * this_void = (char *) this;

		this_void = this_void + lock_and_size.as_bitfield.size - 8;

		return (uint64_t * ) this_void;

	}

	__device__ void set_footer(uint64_t num_bytes){

		uint64_t * footer = get_footer();
		footer[0] = num_bytes;
	}

	// uint64_t read_prev_footer_atomic(){

	// 	char * char_this = (char *) this;
	// 	uint64_t * address = (uint64_t *) (this-8);

	// 	atomicCAS((unsigned long long int *) address, 0ULL, 0ULL);

	// }

	__device__ void set_size(uint64_t new_size){

		lock_and_size.as_bitfield.size = new_size;
		return;
	}

	__device__ uint64_t get_size(){
		return this->lock_and_size.as_bitfield.size;
	}

	__device__ void set_next(header * next_node){

		assert(this != nullptr);
		next_ptr = next_node;
	}

	__device__ void set_prev(header * prev_node){

		assert(this != nullptr);
		prev_ptr = prev_node;
	}


	__host__ __device__ header * get_next(){
		return next_ptr;
	}

	__host__ __device__ header * get_prev(){
		return prev_ptr;
	}


	__device__ static header * init_node(void * allocation, uint64_t num_bytes){

		header * node = (header *) allocation;

		node->lock_and_size.as_bitfield.lock = 0;

		__threadfence();

		node->stall_lock();

		char * footer_address = ((char *) allocation) + num_bytes - 8;

		uint64_t * footer = (uint64_t *) footer_address;

		footer[0] = num_bytes;



		
		node->lock_and_size.as_bitfield.size = num_bytes;

		//printf("Size %llu\n", node->lock_and_size.as_bitfield.size);

		return node;

	}

	__host__ static header * init_heap(uint64_t num_bytes){

		void * allocation;

		cudaMalloc((void **)&allocation, num_bytes);

		init_heap_kernel<header><<<1,1>>>(allocation, num_bytes);

		return (header *) allocation;

	}

	__host__ static header * init_heap_managed(uint64_t num_bytes){

		void * allocation;

		cudaMallocManaged((void **)&allocation, num_bytes);

		init_heap_kernel<header><<<1,1>>>(allocation, num_bytes);

		return (header *) allocation;

	}

	__host__ static void free_heap(header * heap){

		void * allocation = (void *) heap;

		cudaFree(allocation);

	}

	__device__ void static init_heap_global(void * unsigned_allocation, uint64_t num_bytes){


		char * allocation = (char *) unsigned_allocation;

		header * heap_start = header::init_node(allocation, 48);
		header * main_node = header::init_node(allocation+48, num_bytes-96);
		header * heap_end = header::init_node(allocation+num_bytes-48, 48);


		heap_start->lock_and_size.as_uint ^= ENDPOINT_MASK;
		heap_end->lock_and_size.as_uint ^= ENDPOINT_MASK;

		heap_start->set_next(main_node);
		main_node->set_next(heap_end);
		heap_end->set_next(heap_start);

		heap_start->set_prev(heap_end);
		main_node->set_prev(heap_start);
		heap_end->set_prev(main_node);

		#if DEBUG_PRINTS
		printf("%llu, %llu, %llu\n", heap_start->get_size(), main_node->get_size(), heap_end->get_size());

		//header_to_return[0] = heap_start

		heap_start->printnode();
		main_node->printnode();
		heap_end->printnode();
		#endif

		heap_start->unlock();
		main_node->unlock();
		heap_end->unlock();


	}


	//returns true if block has at least size bytes available aligned to alignment - offset
	__device__ bool has_valid_split(uint64_t size, uint64_t alignment, int offset){


		assert(__popc(alignment) == 1);

		uint64_t block_size = this->get_size();

		uint64_t ptr = (uint64_t) this;

		uint64_t valid_offset = (ptr % alignment) + offset;

		if (valid_offset + size <= block_size){
			return true;
		}


		return false;





	}

	//create a new node with num_bytes space
	//this includes the header / footer in num_bytes, as is convention for this library
	//not really non locking at the moment [TODO]
	//so the name is probably confusing if you're not me :D
	__device__ header * split_non_locking(uint64_t num_bytes){



		//32 + 1000 + 16 bytes
		//size is 1048
		//request 500

		//32 + 500 + 16

		//32 + 4-- + 16


		//atomicLock myLock(&lock_and_size);

		//header * next_node = get_next();

		//atomicLock nextLock(&next_node->lock_and_size);


		lock_and_size.as_bitfield.size -= num_bytes;

		set_footer(lock_and_size.as_bitfield.size);

		//we pad with new footer as our footer is absorbed by the next init_node
		char * next_address = ((char *) this) + lock_and_size.as_bitfield.size;

		//init_node now implicitly locks
		header * new_node = init_node((void *) next_address, num_bytes);

		//no need to do this if we immediately remove from list

		// this->set_next(new_node);

		// new_node->set_next(next_node);

		// new_node->set_prev(this);

		// next_node->set_prev(new_node);

		return new_node;


	}

	__device__ void remove_from_list(){

		header * prev = get_prev();

		header * next = get_next();

		#if DEBUG_ASSERTS

		assert(prev != next);

		#endif

		prev->set_next(next);

		next->set_prev(prev);

		return;

	}

	//merge this into prev
	//requires both locked 
	//prev_in_mem remains locked
	__device__ void merge(header * prev_in_mem){


		//can't remove from list twice
		//remove_from_list();

		uint64_t new_size = get_size()+prev_in_mem->get_size();

		prev_in_mem->set_size(new_size);
		prev_in_mem->set_footer(new_size);

		return;


		 

	}

	__device__ bool inline lock_node_and_neighbors(){

		if (lock()){

			if (lock_neighbors()){
				return true;
			}

			unlock();

		}

		return false;



	}

	//locks this, this->next, this->next->next
	__device__ bool inline lock_node_and_next_two(){

		if (lock()){

			header * next = get_next();



			if (next->lock()){


				header * next_next = next->get_next();

				if (next_next == this) return true;

				if (next_next->lock()){
					return true;
				}


				//printf("Next next node available\n");

				next->unlock();
			} else {
				//printf("Next node available\n");
			}


			unlock();


		} else {
			//printf("Head not available\n");
		}

		return false;

	}


	__device__ bool inline lock_node_and_next(){

		if (lock()){

			header * next = get_next();



			if (next->lock()){

				return true;

			} else {
				//printf("Next node unavailable\n");
			}


			unlock();


		} else {
			//printf("Head not available\n");
		}

		return false;

	}


	__device__ bool inline lock_neighbors(){

		header * prev = get_prev();

		header * next = get_next();

		if (prev == next){

			if (prev->lock()){
				return true;
			} else {
				return false;
			}

		}

		if (prev->lock()){

			if (next->lock()){

				return true;

			}

			prev->unlock();
		}

		return false;



	}


	__device__ void unlock_neighbors(){


		header * prev = get_prev();

		header * next = get_next();

		if (prev == next){
			prev->unlock();
		} else {
			prev->unlock();
			next->unlock();
		}



	}


	//attempt to lock the node ahead of me
	__device__ header * try_get_prev_node_locked(){

		#if DEBUG_ASSERTS

		assert(!check_endpoint());

		#endif


		//shift exactly 8 bytes back
		uint64_t * prev_footer = ((uint64_t *) this) -1;

	
		uint64_t alt_size = atomicCAS((unsigned long long int *) prev_footer, 0ULL,0ULL);

		char * header_address = ((char * ) this) - alt_size;

		header * prev_header = (header *) header_address;

		if (prev_header->lock()){
			return prev_header;
		}

		return nullptr;


	}

	//attempt to lock the node ahead of me
	//if we already own it don't modify locks
	__device__ header * try_get_prev_node(header * heap, header * next){

		#if DEBUG_ASSERTS

		assert(!check_endpoint());

		#endif


		//shift exactly 8 bytes back
		uint64_t * prev_footer = ((uint64_t *) this) -1;

	
		uint64_t alt_size = atomicCAS((unsigned long long int *) prev_footer, 0ULL,0ULL);

		char * header_address = ((char * ) this) - alt_size;

		header * prev_header = (header *) header_address;

		if (heap == prev_header || next == prev_header){
			return prev_header;
		}

		if (prev_header->lock()){
			return prev_header;
		}

		return nullptr;


	}

	__device__ header * get_prev_mem_node(){

		uint64_t * prev_footer = ((uint64_t *) this) -1;

	
		uint64_t alt_size = atomicCAS((unsigned long long int *) prev_footer, 0ULL,0ULL);

		char * header_address = ((char * ) this) - alt_size;

		header * prev_header = (header *) header_address;

		return prev_header;

	}

	//returns the next node
	//state is unknown
	__device__ header * get_next_node_in_mem(){

		char * header_address = ((char *) this) + get_size();

		header * next_header = (header *) header_address;

		return next_header;

	}

	//BUG: the head node (e00000) is locked
	__device__ void merge_next_if_available(){


		header * next = get_next_node_in_mem();

		//change structure of this
		//stall lock IFF




		//one liner to establish lock
		while (!next->lock_node_and_neighbors());





		//now that next node is secure 
		if ((!next->check_endpoint()) && (!next->check_alloc())){

			//free_to_merge
			next->remove_from_list();
			next->unlock_neighbors();

			next->merge(this);

		} else {

			//couldn't merge, forget it
			next->unlock_neighbors();
			next->unlock();

		}

	}

	__device__ static header * get_header_from_address(void * uncasted_address){

		char * address = ((char * ) uncasted_address) - 32;

		header * head = (header *) address ;

		return head;

	}

	__device__ header * merge_nodes(header * prev, header * next){


		uint64_t total_size = prev->get_size() + next->get_size();

		assert(prev->get_next_node_in_mem() == next);
		assert(next->get_prev_mem_node() == prev);

		prev->set_size(total_size);
		prev->set_footer(total_size);

		assert(prev->get_footer()[0] == total_size);

		return prev;

	}

	__device__ void free(void * uncasted_address){
		free_safe(uncasted_address);
	}


	__device__ void free_safe(void * uncasted_address){

		header * head = header::get_header_from_address(uncasted_address);

		


		//atomicLock()
		//head->stall_lock();

		stall_lock();

		head->dealloc();

		//assert changes visible
		__threadfence();

		#if DEBUG_ASSERTS

		assert(check_lock());
		
		#endif

		header * prev_node = head->get_prev_mem_node();

		bool prev_detached = false;

		if ((!prev_node->check_alloc()) && (!prev_node->check_endpoint())){

			prev_detached = true;
			prev_node->remove_from_list();

		}


		header * next_mem = head->get_next_node_in_mem();

		bool next_detached = false;

		if ((!next_mem->check_alloc()) && (!next_mem->check_endpoint())){


			next_mem->remove_from_list();

			next_detached = true;

		}


		//printf("Current state: prev %x")

		if (prev_detached){

			head = merge_nodes(prev_node, head);

		}

		if (next_detached){

			head = merge_nodes(head, next_mem);

		}

		//head now set to new size


		header * next = get_next();

		this->set_next(head);

		head->set_next(next);

		head->set_prev(this);

		next->set_prev(head);




		__threadfence();

		unlock();

		return;


	}



	// __device__ void * find_first_fit(uint64_t num_bytes){

	// 	//printf("Starting ")

	// 	header * prev = this;

	// 	prev->stall_lock();

	// 	header * end_node = prev->get_prev();

	// 	header * main = get_next();

	// 	main->stall_lock();

	// 	header * next = main->get_next();

	// 	next->stall_lock();

	// 	printf("Nodes locked\n");

	// 	while (next != this){

	// 		if (main->get_size() >= num_bytes){



	// 			//detach
	// 			uint64_t leftover = main->get_size() - num_bytes;

	// 			if (leftover > CUTOFF_SIZE && leftover > 48){

	// 				printf("Splitting %llu, need %llu bytes\n", main->get_size(), num_bytes);

	// 				header * ideal_node = main->split_non_locking(num_bytes);

	// 				//ideal_node->remove_from_list();

	// 				prev->unlock();
	// 				main->unlock();
	// 				next->unlock();

	// 				ideal_node->alloc();
	// 				ideal_node->unlock();

	// 				char * ideal_address = ((char *) ideal_node) + 32;


	// 				return (void *) ideal_address;


	// 			} else {

	// 				printf("Need %llu bytes, just freeing whole node\n");

	// 				main->printnode();

	// 				main->remove_from_list();

	// 				main->alloc();

	// 				prev->unlock();
	// 				main->unlock();
	// 				next->unlock();

	// 				char * ideal_address = ((char *) main) + 32;

	// 				return (void *) ideal_address;



	// 			}

	// 		} else {

	// 			//locking scheme problem?

	// 			//3 or 4 nodes

	// 			prev->unlock();
	// 			//header * next_next = stall_lock();


	// 		}

	// 	}

	// 	prev->unlock();
	// 	main->unlock();
	// 	next->unlock();
	// 	return nullptr;


	// }


	__device__ void * find_first_fit_end_node(uint64_t num_bytes){

		//printf("Starting ")

		while(!lock_node_and_next_two()){

			//printf("Stalling acquiring all nodes\n");

		}

		header * prev = this;

		#if DEBUG_ASSERTS
		printf("Malloc debugs are on\n");

		assert(prev->check_lock());
		#endif

		//prev->stall_lock();

		header * end_node = prev->get_prev();

		#if DEBUG_ASSERTS

		assert(prev->check_endpoint());
		assert(end_node->check_endpoint());

		#endif

		header * main = prev->get_next();

		//main->stall_lock();

		header * next = main->get_next();

		if (next == prev){

			#if DEBUG_PRINTS
			printf("Empty list - returning nullptr!\n");
			#endif

			__threadfence();
			main->unlock();
			prev->unlock();

			return nullptr;
		}


		#if DEBUG_ASSERTS

		assert(prev->check_lock());
		assert(main->check_lock());
		assert(next->check_lock());
		assert (prev != next);

		#endif

		#if DEBUG_PRINTS
		printf("Nodes locked\n");
		#endif

		while (true){


			if (main->get_size() >= num_bytes){



				//detach
				uint64_t leftover = main->get_size() - num_bytes;

				if (leftover > CUTOFF_SIZE && leftover > 48){

					printf("Splitting %llu, need %llu bytes\n", main->get_size(), num_bytes);

					header * ideal_node = main->split_non_locking(num_bytes);

					//ideal_node->remove_from_list();

					__threadfence();
					ideal_node->alloc();
					ideal_node->unlock();

					prev->unlock();
					main->unlock();
					next->unlock();



					char * ideal_address = ((char *) ideal_node) + 32;


					return (void *) ideal_address;


				} else {

					#if DEBUG_PRINTS
					printf("Need %llu bytes, just freeing whole node\n", num_bytes);
					#endif

					main->printnode();

					main->remove_from_list();

					main->alloc();

					prev->unlock();
					main->unlock();
					next->unlock();

					char * ideal_address = ((char *) main) + 32;

					return (void *) ideal_address;



				}

			} else {

				//locking scheme problem?

				//3 or 4 nodes

				if (next == end_node){


					printf("Malloc failure\n");
					prev->unlock();
					main->unlock();
					next->unlock();
					return nullptr;



				} else {

					printf("Shifting\n");

					header * next_next = next->get_next();

					prev->unlock();
					
					next_next->stall_lock();

					prev = main;
					main = next;
					next = next_next;


				}



			}



		}




	}

	__device__ void * find_first_safe(uint64_t num_bytes){

		//printf("Starting ")
		stall_lock();

		header * prev = this;

		//prev->stall_lock();

		header * end_node = prev->get_prev();

		header * main = prev->get_next();

		header * next = main->get_next();

		if (next == prev){

			//printf("Empty list - returning nullptr!\n");

			__threadfence();

			unlock();
			return nullptr;
		}

		// uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;
		// printf("%llu Safe Locks secured\n", tid);

		//printf("%llu Safe Locks secured\n", threadIdx.x+blockIdx.x*blockDim.x);

		#if DEBUG_ASSERTS

		assert(prev->check_lock());
		assert (prev != next);

		#endif

		//printf("Nodes locked\n");

		while (true){


			if (main->get_size() >= num_bytes){



				//detach
				uint64_t leftover = main->get_size() - num_bytes;

				if (leftover > CUTOFF_SIZE && leftover > 48){

					//printf("Splitting %llu, need %llu bytes\n", main->get_size(), num_bytes);

					header * ideal_node = main->split_non_locking(num_bytes);

					//ideal_node->remove_from_list();



					__threadfence();

					ideal_node->alloc();

					ideal_node->unlock();

					unlock();



					char * ideal_address = ((char *) ideal_node) + 32;


					return (void *) ideal_address;


				} else {

					//printf("Need %llu bytes, just freeing whole node\n", num_bytes);

					main->printnode();

					main->remove_from_list();

					main->alloc();

					unlock();


					char * ideal_address = ((char *) main) + 32;

					return (void *) ideal_address;



				}

			} else {

				//locking scheme problem?

				//3 or 4 nodes

				if (next == end_node){


					//printf("Malloc failure\n");
					unlock();
					return nullptr;



				} else {

					//printf("Shifting\n");

					header * next_next = next->get_next();

					prev = main;
					main = next;
					next = next_next;


				}



			}



		}




	}



	
	__device__ void * find_first_safe_aligned(uint64_t num_bytes, uint64_t alignment, int offset){

		//printf("Starting ")
		stall_lock();

		header * prev = this;

		//prev->stall_lock();

		header * end_node = prev->get_prev();

		header * main = prev->get_next();

		header * next = main->get_next();

		if (next == prev){

			//printf("Empty list - returning nullptr!\n");

			__threadfence();

			unlock();
			return nullptr;
		}

		// uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;
		// printf("%llu Safe Locks secured\n", tid);

		//printf("%llu Safe Locks secured\n", threadIdx.x+blockIdx.x*blockDim.x);

		#if DEBUG_ASSERTS

		assert(prev->check_lock());
		assert (prev != next);

		#endif

		//printf("Nodes locked\n");


		//split into cases
		//main is an exact match or the leftover is too small to bother with

		//main is a match that is too long


		//main is not a match but could produce a node that is
		// will split into three nodes, including the ideal
			//will split into two nodes if too small


		//main is not a match and does not have the space for a match
		//this includes the case of 

		while (true){


			//this calculation checks if any node has the potential for alignment
			//if this node matches perfectly it should be ok

			//special case? if head += 32 is aligned then we're ok



			uint64_t main_as_uint = (uint64_t) main;

			//how many units above the alignment is main?
			uint64_t distance_above_alignment = (main_as_uint + 32 + offset) % alignment;

			if (distance_above_alignment == 0){


				//exact match
				if (main->get_size() >= num_bytes){

					uint64_t leftover = main->get_size() - num_bytes;

					
					if (leftover > CUTOFF_SIZE && leftover > 48){


						//case 1
						//we are already aligned! we just need to siphon off extra space somewhere else

						//ideal node takes the rest of the space
						header * remainder_node = main->split_non_locking(leftover);

						//remainder_node->stall_lock();

						prev->set_next(remainder_node);

						next->set_prev(remainder_node);

						remainder_node->set_prev(prev);
						remainder_node->set_next(next);

						main->alloc();
						
						//maybe
						//main->unlock();

						//next->unlock();
						remainder_node->unlock();
						unlock();



						char * ideal_address = ((char *) main) + 32;

						#if DEBUG_ASSERTS


						uint64_t ideal_as_uint = (uint64_t) ideal_address;

						assert((ideal_as_uint + offset) % alignment == 0);

						#endif

						return (void *) ideal_address;

						//return main;

					} else {

						//case 2
						//there is enough space for this node, but not enough to generate a leftover

						main->remove_from_list();

						main->alloc();

						//next->unlock();
						//main->unlock();
						unlock();

						char * ideal_address = ((char *) main) + 32;

						#if DEBUG_ASSERTS


						uint64_t ideal_as_uint = (uint64_t) ideal_address;

						assert((ideal_as_uint + offset) % alignment == 0);

						#endif

						return (void *) ideal_address;





					}

					//header * ideal_node = main->split_non_locking(leftover);

				}


			}

			//how many more bytes do we need to traverse?
			uint64_t distance_to_next = alignment - distance_above_alignment;

			//if less than 48 bytes, we need to move to the next alignment address
			//that is both aligned and >= 48 bytes

			while (distance_to_next < 48){
				distance_to_next += alignment;
			}

			//distance to next is the distance from the main ptr to the aligned address
			//we we need to subtract 32 bytes

			if (distance_to_next + num_bytes <= main->get_size()){


				//main starts at 0
				//next node should be distance_to_next
				//node starts 32 back!
				//distance to next starts from the data of main, not the head
				//so no subtraction required
				uint64_t bytes_in_main = distance_to_next;


				uint64_t bytes_to_allocate = main->get_size() - bytes_in_main;

				header * half_node = main->split_non_locking(bytes_to_allocate);

				#if DEBUG_ASSERTS

				uint64_t half_as_uint = (uint64_t) half_node;
				assert(half_as_uint % alignment == 0);
				assert(half_node->get_size() >=num_bytes);
				#endif


				if (bytes_to_allocate - num_bytes > 48){
					//split into a second node

					uint64_t remaining_bytes = bytes_to_allocate - num_bytes;

					header * final_node = half_node->split_non_locking(remaining_bytes);

					main->set_next(final_node);
					final_node->set_next(next);

					next->set_prev(final_node);
					final_node->set_prev(main);

					half_node->alloc();

					//next->unlock();
					final_node->unlock();
					half_node->unlock();
					//main->unlock();

					__threadfence();
					unlock();

					char * ideal_address = ((char *) half_node) + 32;

					#if DEBUG_ASSERTS


					uint64_t ideal_as_uint = (uint64_t) ideal_address;

					assert((ideal_as_uint + offset) % alignment == 0);

					#endif

					return (void *) ideal_address;


				} else {

					//return the half node

					half_node->alloc();
					half_node->unlock();
					__threadfence();
					unlock();

					char * ideal_address = ((char *) half_node) + 32;

					#if DEBUG_ASSERTS


					uint64_t ideal_as_uint = (uint64_t) ideal_address;

					assert((ideal_as_uint + offset) % alignment == 0);

					#endif

					return (void *) ideal_address;
					//return half_node;


				}

			}


			//locking scheme problem?

			//3 or 4 nodes

			if (next == end_node){


				//printf("Malloc failure\n");
				unlock();
				return nullptr;



			} else {

				//printf("Shifting\n");

				header * next_next = next->get_next();

				prev = main;
				main = next;
				next = next_next;


			}






		}




	}


	__device__ void * malloc_safe(uint64_t bytes_requested){

		//bytes of an object contains header + footer

		#if COUNTING_CYCLES
		uint64_t heap_counter_start = clock64();
		#endif



		bytes_requested += 48;

		void * my_malloc = find_first_safe(bytes_requested);

		#if COUNTING_CYCLES
			
			uint64_t heap_counter_end = clock64();

			uint64_t free_list_total_time = (heap_counter_end - heap_counter_start)/COMPRESS_VALUE;

			atomicAdd((unsigned long long int *) &free_list_counter, (unsigned long long int) free_list_total_time);

			atomicAdd((unsigned long long int *) &heap_traversals, (unsigned long long int) 1);


		#endif

		return my_malloc;

	}

	__device__ void * malloc_aligned(uint64_t bytes_requested, uint64_t alignment, int offset){

		//bytes of an object contains header + footer
		bytes_requested += 48;

		//minimum alignment is 16 bytes

		assert(alignment % 16 == 0);
		assert(offset % 16 == 0);

		#if COUNTING_CYCLES
			uint64_t heap_counter_start = clock64();
		#endif

		void * my_malloc = find_first_safe_aligned(bytes_requested, alignment, offset);

		#if COUNTING_CYCLES
			
			uint64_t heap_counter_end = clock64();

			uint64_t free_list_total_time = (heap_counter_end - heap_counter_start)/COMPRESS_VALUE;

			atomicAdd((unsigned long long int *) &free_list_counter, (unsigned long long int) free_list_total_time);

			atomicAdd((unsigned long long int *) &heap_traversals, (unsigned long long int) 1);


		#endif

		return my_malloc;

	}

	__host__ void * host_malloc_aligned(uint64_t bytes_requested, uint64_t alignment, int offset){

		void ** temp_write_space;
		cudaMallocManaged(&temp_write_space, sizeof(void *));

		if (temp_write_space == nullptr){
			printf("No space for cudaMalloc\n");
			abort();
		}

		host_malloc_aligned_kernel<header><<<1,1>>>(this, temp_write_space, bytes_requested, alignment, offset);
		cudaDeviceSynchronize();

		void * final_address = temp_write_space[0];
		cudaFree(temp_write_space);

		return final_address;

	}



	__device__ void * malloc(uint64_t bytes_requested){

		//bytes of an object contains header + footer
		return malloc_safe(bytes_requested);

	}


	//debug funcs
	__device__ static void print_heap(header * head){

		header * loop_ptr = head;

		printf("Current state of free nodes:\n");

		loop_ptr->printnode();
		loop_ptr = loop_ptr->get_next();

		while (loop_ptr != head){
			loop_ptr->printnode();
			loop_ptr = loop_ptr->get_next();
		}

	}

 	__device__ void printnode(){

 		uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;
		printf("%llu: Node at %p, size is %llu, next at %p, prev at %p, is head: %d, is_locked: %d, is alloced: %d, footer %llu\n", tid, this, lock_and_size.as_bitfield.size, get_next(), get_prev(), check_endpoint(), check_lock(), check_alloc(), get_footer()[0]);
	}

	__device__ void generate_report(reporter * my_reporter){

		header * head = this;

		header * loop_ptr = this;

		//print_heap(this);

		uint64_t largest = 0;

		uint64_t free_size = loop_ptr->get_size();

		loop_ptr = loop_ptr->get_next();

		while (loop_ptr != head){

			free_size += loop_ptr->get_size();

			if (loop_ptr->get_size() > largest){
				largest = loop_ptr->get_size();
			}
			loop_ptr = loop_ptr->get_next();



		}

		uint64_t total_size =  ((uint64_t) head->get_prev()) - ((uint64_t) head) + 48;

		// printf("Head %llu\n", (uint64_t) head);
		// printf("Head prev %llu\n", ((uint64_t) head->get_prev()));

		// printf("Total size\n")

		my_reporter->modify_heap_bytes_total(total_size);
		my_reporter->modify_heap_bytes_free(free_size);
		my_reporter->modify_fragmentation_count(largest);

	}

		

};


__device__ inline atomicLock::atomicLock(header * head){

}

__device__ inline atomicLock::~atomicLock(){

		 header_to_lock->unlock();

}


// struct atomicLock{

// 	header_bitfield * header_to_lock;

// 	// __device__ inline atomicLock(header_bitfield * ext_bitfield){

// 	// 	header_to_lock = ext_bitfield;

// 	// 	header_to_lock->stall_lock();

// 	// }

// 	__device__ inline atomicLock(header * head){

// 		header_to_lock = head->get_lock_and_size_addr();

// 		head->stall_lock();

// 	}

// 	__device__ inline ~atomicLock(){

// 		 header_to_lock->unlock();

// 	}

// };



}

}


#endif //GPU_BLOCK_