#ifndef POGGERS_BIT_SLAB
#define POGGERS_BIT_SLAB


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>


#define SLAB_PRINT_DEBUG 0


namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 



__device__ bool check_indices(cg::coalesced_group & active_threads, int my_index){

	uint64_t my_mask;

	if (my_index == -1){
		
		my_mask = 0;

	} else {

		my_mask = SET_BIT_MASK(my_index);

	}


	

	uint64_t scanned_mask = cg::exclusive_scan(active_threads, my_mask, cg::bit_or<uint64_t>());

	if (my_index == -1){
		return true;
	}

	// if (my_index == -1){
	// 	printf("the fuck\n");
	// }

	#if SLAB_PRINT_DEBUG
	printf("Thread %d has item %d, bits scanned_mask & mask: %llx %llx\n", threadIdx.x, my_index, scanned_mask, my_mask);
	#endif
	//scan mask should not include me - and should be 0.
	return !(scanned_mask & my_mask);

}

//global helper 
//make this select -1 if not grabbable
//does modify
__device__ int select_unique_bit_int(cg::coalesced_group & active_threads, uint64_t_bitarr active_bits){


		int my_bit = -1;

		// if (active_threads.thread_rank() == 0){
		// 	printf("Starting up selection process with active bits %lu\n", copy_active_bits.bits);
		// }


		while (true){

			cg::coalesced_group searching_group = cg::coalesced_threads();

			int bit = active_bits.get_random_active_bit();

			//if not available fucken crash
			if (bit == -1){ my_bit = -1; break; }

			uint64_t my_mask = (1ULL) << bit;

			//now scan across the masks
			uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

			//final thread needs to broadcast updates
			if (searching_group.thread_rank() == searching_group.size()-1){

				//doesn't matter as the scan only adds bits
				//not to set the mask to all bits not taken
				uint64_t final_mask = ~(scanned_mask | my_mask);

				active_bits.apply_mask(final_mask);

			}


			//everyone now has an updated final copy of ext bits?
			active_bits = searching_group.shfl(active_bits, searching_group.size()-1);

			//most up to date changes are here - not propogating.

			if (!(scanned_mask & my_mask)){

				//I received an item!
				//allocation has already been marked and index is set
				//break to recoalesce for exit
				my_bit = bit;
				break;



			}





		}

		//everyone needs to ballot on what the smallest version of bits is.
		int min = cg::reduce(active_threads, __popcll(active_bits), cg::less<int>());

	 	int leader = __ffs(active_threads.ballot(__popcll(active_bits) == min));



		//group needs to synchronize
		active_bits = active_threads.shfl(active_bits, leader);

		return my_bit;

}




//global helper 
//make this select -1 if not grabbable
//does modify
__device__ int select_unique_bit(cg::coalesced_group & active_threads, uint64_t_bitarr & active_bits){


		int my_bit = -1;

		// if (active_threads.thread_rank() == 0){
		// 	printf("Starting up selection process with active bits %lu\n", copy_active_bits.bits);
		// }


		while (true){

			cg::coalesced_group searching_group = cg::coalesced_threads();

			#if SLAB_PRINT_DEBUG
			if (searching_group.thread_rank() == 0)
			printf("Start of round: %d in group, bits %llx\n", searching_group.size(), active_bits.bits);
			#endif


			int bit = active_bits.get_random_active_bit();

			//if not available fucken crash
			if (bit == -1){ my_bit = -1; break; }

			uint64_t my_mask = (1ULL) << bit;


			//now scan across the masks
			uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

			//final thread needs to broadcast updates
			if (searching_group.thread_rank() == searching_group.size()-1){

				//doesn't matter as the scan only adds bits
				//not to set the mask to all bits not taken
				uint64_t final_mask = ~(scanned_mask | my_mask);

				active_bits.apply_mask(final_mask);

				#if SLAB_PRINT_DEBUG
				printf("Team member %d/%d sees new bits as %llx, new mask %llx, popcount of bits %d\n", searching_group.thread_rank(), searching_group.size(), active_bits, scanned_mask, __popcll(active_bits.bits));
				#endif

			}

			//everyone now has an updated final copy of ext bits?
			active_bits = searching_group.shfl(active_bits, searching_group.size()-1);

			if (!(scanned_mask & my_mask)){

				//I received an item!
				//allocation has already been marked and index is set
				//break to recoalesce for exit
				my_bit = bit;
				break;



			}



		}


		//group needs to synchronize
		//everyone needs to ballot on what the smallest version of bits is.
		int min = cg::reduce(active_threads, __popcll(active_bits), cg::less<int>());


		#if SLAB_PRINT_DEBUG
		printf("Min is %d, my_popcount active_bits %llx\n", min, active_bits);
		#endif

	 	int leader = __ffs(active_threads.ballot(__popcll(active_bits) == min))-1;



		//group needs to synchronize
		active_bits = active_threads.shfl(active_bits, leader);





		return my_bit;



		// uint64_t my_mask;

		// //people who don't get an index don't get to set mask to 11111111...
		// if (my_bit == -1){

		// 	#if SLAB_PRINT_DEBUG
		// 	if (__popcll(active_bits.bits) >= active_threads.size()){

		// 		printf("Bug is in selecting indices\n");

		// 	}

		// 	#endif

		// 	my_mask = 0;
		// } else {
		// 	my_mask = (1ULL << my_bit);
		// }
		

		// uint64_t scanned_mask = cg::exclusive_scan(active_threads, my_mask, cg::bit_or<uint64_t>());

		// if (active_threads.thread_rank() == active_threads.size()-1){


		// 	uint64_t final_mask = ~(scanned_mask | my_mask);

		// 	active_bits.apply_mask(final_mask);


		// }

		// active_bits = active_threads.shfl(active_bits, active_threads.size()-1);

		// return my_bit;



}



struct warp_lock {

	uint64_t_bitarr lock_bits;

	__device__ void init(){

		lock_bits = 0ULL;

	}

	__device__ int get_warp_bit(){

		return (threadIdx.x / 32);

	}

	__device__ bool lock(){

		return lock_bits.set_bit_atomic(get_warp_bit());

	}

	__device__ void unlock(){

		lock_bits.unset_bit_atomic(get_warp_bit());

	}

	__device__ void spin_lock(){


		while (!lock());

	}

};



struct offset_alloc_bitarr{

	uint64_t internal_offset;
	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];

	__device__ void init(){

		manager_bits.bits = ~(0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = ~(0ULL);
		}
		//at some point work on this
		internal_offset = ~0ULL;

	}


	__device__ void attach_allocation(uint64_t ext_offset){

		internal_offset = ext_offset;

		if ((internal_offset % 4096) != 0){
			printf("Logical bug in bucket reasoning! bits up to 4096 not actually free\n");
		}

	}

	

	//TODO: templatize over size in bytes

	//bug in here releasing small allocs
	// __device__ bool free_allocation(uint64_t offset){


	// 	//safety check - cast into my setup
	// 	int upper_bit = (offset-(internal_offset & ~1ULL ) )/64;

	// 	int lower_bit = (offset-(internal_offset & ~1ULL ))% 64; 

	// 	if (upper_bit > 63 || lower_bit > 63){
	// 		printf("Free bug - upper %d lower %d\n", upper_bit, lower_bit);
	// 	}


	// 	//collate thread teams together
	// 	while (true){

	// 		cg::coalesced_group active_threads = cg::coalesced_threads();

	// 		//only threads that match with the leader may progress.

	// 		int team_upper_bit = active_threads.shfl(upper_bit, 0);

	// 		if (team_upper_bit == upper_bit) break;

			

	// 	}

	// 	//starting team now shares the same upper bit
	// 	cg::coalesced_group starting_team = cg::coalesced_threads();


	// 	uint64_t my_mask = (1ULL << lower_bit);

	// 	uint64_t scanned_mask = cg::inclusive_scan(starting_team, my_mask, cg::bit_or<uint64_t>());

	// 	if (starting_team.thread_rank() == starting_team.size()-1){

			
	// 		uint64_t old_bits = alloc_bits[upper_bit].set_OR_mask(scanned_mask);

	// 		#if SLAB_PRINT_DEBUG
	// 		if ((old_bits & scanned_mask) != 0ULL){
	// 			printf("%d Bug in overwriting bits, %d --  %llx overwritten\n", threadIdx.x/32, upper_bit, old_bits & scanned_mask);
	// 		}
	// 		#endif

	// 		if ((old_bits | scanned_mask) == (~0ULL)){

	// 			uint64_t old = atomicExch((unsigned long long int *)&alloc_bits[upper_bit], ~0ULL);

	// 			#if SLAB_PRINT_DEBUG
	// 			if (old != ~0ULL){
	// 				printf("%d Free bug with %d - old is %llx\n", threadIdx.x/32, upper_bit, old);
	// 			}
	// 			#endif

	// 			__threadfence();

	// 			if (manager_bits.set_index(upper_bit) & SET_BIT_MASK(upper_bit)){

	// 				#if SLAB_PRINT_DEBUG
	// 				printf("failed to reclaim bit %d\n", upper_bit);
	// 				#endif
	// 				return false;				

	// 			} else {


	// 				#if SLAB_PRINT_DEBUG
	// 				printf("Returned %d\n", upper_bit);
	// 				#endif

	// 				return true;

	// 			}

				


	// 		}

	// 	}

	// 	return true;


	// }

	//helper functions

	//frees must succeed - precondition - fail on double free but print error.
	__device__ bool free_allocation_v2(uint64_t offset){


		int upper_bit = (offset-(internal_offset & ~1ULL))/64;

		int lower_bit = (offset-(internal_offset & ~1ULL)) % 64; 


		if (upper_bit > 63 || lower_bit > 63){
			printf("Free bug - upper %d lower %d\n", upper_bit, lower_bit);
		}



		//uint64_t my_mask = (1ULL << lower_bit);

		uint64_t old = alloc_bits[upper_bit].set_index(lower_bit);

		if (old & SET_BIT_MASK(lower_bit)){
			#if SLAB_PRINT_DEBUG
			printf("Double free bug\n");
			#endif
			return false;
		}

		if (__popcll(old) == 63){


			old = atomicExch((unsigned long long int *)&alloc_bits[upper_bit], ~0ULL);

			#if SLAB_PRINT_DEBUG
			if (old != ~0ULL){
				printf("Bug in conversion\n");
			}
			#endif

			uint64_t old_bits = manager_bits.set_index(upper_bit);

			if (old_bits & SET_BIT_MASK(upper_bit)){

				#if SLAB_PRINT_DEBUG
				printf("failed to reclaim bit %d\n", upper_bit);
				#endif
				return false;				

			} else {


				#if SLAB_PRINT_DEBUG
				printf("Returned %d\n", upper_bit);
				#endif
				return __popcll(old_bits) == 63;

			}


		}


		return false;



	}

	__device__ bool is_full_atomic(){


		return __popcll(atomicOr((unsigned long long int *)&manager_bits, 0ULL)) == 64;

	}


	__device__ bool bit_malloc_v2(cg::coalesced_group & active_threads, uint64_t & offset, uint64_t & remainder){


		// if (active_threads.thread_rank() == 0){
		// 	printf("Warp %d has entered the selection process!\n", threadIdx.x/32);
		// }


		#if SLAB_GLOBAL_LOADING

		manager_bits.global_load_this();

		#endif

		int upper_index;

		while (active_threads.thread_rank() == 0){

			upper_index = manager_bits.get_random_active_bit();

			if (upper_index == -1) break;


			if (manager_bits.unset_index(upper_index) & SET_BIT_MASK(upper_index)) break;

			// if (active_threads.thread_rank() == 0){
			// 	printf("Warp %d has failed to claim index %d!\n", threadIdx.x/32, upper_index);
			// }

			//printf("Stuck in main bit_malloc_v2\n");

		}


		upper_index = active_threads.shfl(upper_index, 0);

		if (upper_index == -1) return false;

		#if SLAB_PRINT_DEBUG
		if (active_threads.thread_rank() == 0){
			printf("Warp %d has claimed index %d!\n", threadIdx.x/32, upper_index);
		}
		#endif

		uint64_t_bitarr bits;

		//leader performs full swap
		if (active_threads.thread_rank() == 0){

			bits = alloc_bits[upper_index].swap_to_empty();

		}

		bits = active_threads.shfl(bits, 0);

		#if SLAB_PRINT_DEBUG
		if (active_threads.thread_rank() == 0){

			if (__popcll(bits.bits) < active_threads.size()){
				printf("Warp %d: Not enough allocations: %d for %d threads\n", threadIdx.x/32, __popcll(bits.bits), active_threads.size());
			}

		}
		#endif

		int my_index = select_unique_bit(active_threads, bits);

		#if SLAB_PRINT_DEBUG
		if (!check_indices(active_threads, my_index)){
			printf("Team %d with %d threads, Bug in select unique main alloc index %d\n", threadIdx.x/32, active_threads.size(), my_index);
		}
		#endif

		remainder = bits;

		offset = (internal_offset & ~1ULL)+upper_index*64+my_index;

		//this doesn't occur
		if (remainder != 0ULL & my_index == -1){
			printf("Bug in selecting bits\n");
		}

		return (my_index != -1);







	}



	__device__ void mark_pinned(){

		atomicOr((unsigned long long int *)&internal_offset, 1ULL);

	}

	__device__ uint64_t get_offset(){
		return (internal_offset & ~1ULL);
	}

	__device__ bool belongs_to_block(uint64_t offset){


		//printf("Offset: %llu, internal_offset: %llu\n", offset/4096, internal_offset & ~1ULL);

		return (offset)/4096 == (internal_offset & ~1ULL)/4096;

	}


	__device__ void mark_unpinned(){

		atomicAnd((unsigned long long int *)&internal_offset, ~1ULL);
	}


	//returns true when unpinned
	__device__ bool atomic_check_unpinned(){

		return ((poggers::utils::ldca(&internal_offset) & 1ULL) == 0);

	}

	__device__ uint64_t get_active_bits(){

		uint64_t counter = 0;


		for (int i=0; i< 64; i++){
			counter += __popcll(alloc_bits[i]);
		}


		return counter;


	}

	




};



//Correctness precondition
//0000000000000000 is empty key
//if you create it you *will* destroy it
//so other threads don't touch blocks that show themselves as 0ULL
//This allows it to act as the intermediate state of blocks
//and allows the remove pipeline to be identical to above ^
//as we first remove and then re-add if there are leftovers.
struct offset_storage_bitmap{


	//claim bits are 1 if available to claim for store
	uint64_t_bitarr claim_bits;

	//manager bits are 1 if available for malloc
	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];
	uint64_t memmap[64];


	__device__ void init(){


		claim_bits.bits = ~0ULL;
		manager_bits.bits = (0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = (0ULL);
			memmap[i] = ~0ULL;
		}


		

	}


	//move a captured buffer into the pool
	//this has to move in 3 phases
	//1) set buffer pointer from empty
	//2) set remaining allocations
	// __threadfence();
	//3) set manager bit
	__device__ bool attach_buffer(uint64_t ext_buffer, uint64_t ext_bits){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		//uint64_t_bitarr local_copy = claim_bits;

		#if SLAB_PRINT_DEBUG
		printf("Attaching buffer %llu with bits %llx\n", ext_buffer, ext_bits);
		#endif

		while (claim_bits.get_fill() != 0){

			

			
			//printf("Available for claim: %llx\n", claim_bits);


			//allocation_index_bit = local_copy.get_first_active_bit();

			int allocation_index_bit = claim_bits.get_random_active_bit();


	
			//printf("%d: Bit chosen is %d / %llx, %llx %llx\n", threadIdx.x/32, allocation_index_bit, manager_bits, alloc_bits[allocation_index_bit], memmap[allocation_index_bit]);


			//can only swap out if memory is set to 0xffff*8 ...

			if (claim_bits.unset_index(allocation_index_bit) & SET_BIT_MASK(allocation_index_bit)){


				//printf("%d claimed bit %d\n", threadIdx.x/32, allocation_index_bit);


				if (atomicCAS((unsigned long long int *)&memmap[allocation_index_bit], ~0ULL, ext_buffer) == ~0ULL){

						uint64_t swap_bits = alloc_bits[allocation_index_bit].swap_bits(ext_bits);

						if (swap_bits == 0ULL){

							__threadfence();

							if (~(manager_bits.set_index(allocation_index_bit) & SET_BIT_MASK(allocation_index_bit))){

								#if DEBUG_PRINTS
								printf("Manager bit set!\n");
								#endif


								#if SLAB_PRINT_DEBUG
								printf("Allocation bit %d set\n", allocation_index_bit);
								#endif

								return true;

							} else {
								//if you swap out you *must* succeed
								printf("Failure attaching buffer\n");
								assert(1==0);
							}


						} else {
							printf("Memory was set but buffer not empty - This is a bug\n");
							printf("Old memory is %lx\n", swap_bits);
							assert(1==0);
						}


					} else {
						printf("Memmap set failed - failure to properly reset\n");
						assert(1==0);
					}


			}




			//local_copy = manager_bits.global_load_this();
			claim_bits.global_load_this();


		}


		return false;


	}


	//exclusively malloc a section
	//this should lock the section and claim as much as possible.
	//lets just make this cycle! - atomically unset to claim and reset
	//this is important because it lets a team "claim" an allocation - with a gurantee that if they were not satisfied they have now opened a space.
	// with < 64 teams this will always work.
	__device__ bool bit_malloc_v3(cg::coalesced_group & active_threads, uint64_t & offset, uint64_t & remainder){


		#if SLAB_GLOBAL_LOADING
		manager_bits.global_load_this();
		#endif

		//unloading lock bit is a mistake.

		int upper_index;

		while (active_threads.thread_rank() == 0){

			upper_index = manager_bits.get_random_active_bit();

			if (upper_index == -1) break;


			if (manager_bits.unset_index(upper_index) & SET_BIT_MASK(upper_index)) break;

			// if (active_threads.thread_rank() == 0){
			// 	printf("Warp %d has failed to claim index %d!\n", threadIdx.x/32, upper_index);
			// }

			//printf("Stuck in bit malloc secure lock\n");

		}


		upper_index = active_threads.shfl(upper_index, 0);

		if (upper_index == -1) return false;

		//team succeeds or fails together, original team is fine
		alloc_bits[upper_index].global_load_this();

		uint64_t_bitarr bits;
		uint64_t mapping;

		if (active_threads.thread_rank() == 0){
			bits = alloc_bits[upper_index].swap_to_empty();
			mapping = atomicExch((unsigned long long int *)&memmap[upper_index], ~0ULL);

			#if SLAB_PRINT_DEBUG
			printf("Mapping is %llu\n", mapping);
			#endif

		}

		bits = active_threads.shfl(bits,0);
		mapping = active_threads.shfl(mapping, 0);

		int my_index = select_unique_bit(active_threads, bits);

		if (!check_indices(active_threads, my_index)){
			printf("Team %d with %d threads, Bug in select unique main storage index %d\n", threadIdx.x/32, active_threads.size(), my_index);
		}




		remainder = bits;

		offset = mapping + my_index;

		if (active_threads.thread_rank() == 0){
			//atomicExch((unsigned long long int *)&memmap[upper_index], ~0ULL);
			claim_bits.set_index(upper_index);
		}

		return (my_index != -1);

	}


	// __device__ bool bit_malloc(uint64_t & offset){


	// 	//group
	// 	//cg::coalesced_group active_threads = cg::coalesced_threads();

	// 	//team shares the load
	// 	uint64_t_bitarr local_copy = manager_bits.global_load_this();

	// 	#if DEBUG_PRINTS
	// 	if (active_threads.thread_rank() == 0){
	// 		printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
	// 	}
	// 	#endif
		

	// 	while(local_copy.get_fill() != 0ULL){

	// 		cg::coalesced_group active_threads = cg::coalesced_threads();

	// 		int allocation_index_bit = 0;

	// 		//does removing this gate affect performance?

	// 		if (active_threads.thread_rank() == 0){

	// 			//allocation_index_bit = local_copy.get_first_active_bit();

	// 			allocation_index_bit = local_copy.get_random_active_bit();

	// 		}
			
	// 		allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

	// 		uint64_t_bitarr ext_bits;

	// 		bool ballot_bit_set = false;

	// 		if (active_threads.thread_rank() == 0){


	// 			if (manager_bits.unset_bit_atomic(allocation_index_bit)){


	// 				ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

	// 				ballot_bit_set = true;



	// 			}


	// 		}

	// 		//at this point, ballot_bit_set and ext_bits are set in thread 0
	// 		//so we ballot on if we can leave the loop

	// 		if (active_threads.ballot(ballot_bit_set)){


				 
	// 			ext_bits = active_threads.shfl(ext_bits, 0);

	// 			#if DEBUG_PRINTS
	// 			if (active_threads.thread_rank() == 0){
	// 				printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
	// 			}
	// 			#endif


	// 			if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

	// 				//next step: gather threads
	// 				cg::coalesced_group coalesced_threads = cg::coalesced_threads();

	// 				#if DEBUG_PRINTS
	// 				if (coalesced_threads.thread_rank() == 0){
	// 					printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
	// 				}
	// 				#endif

	// 				//how to sync outputs?
	// 				//everyone should pick a random lane?

	// 				//how to coalesce after lanes are picked


	// 				//options
	// 				//1) grab an allocation of the first n and try to  
	// 				//2) select the first n bits ahead of time.

	// 				//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

	// 				//int my_bits = bits_before_index(active_threads.thread_rank());

	// 				// bool ballot = (bits_needeed == my_bits);

	// 				// int result = coalesced_threads.ballot(ballot);

					
	// 				int my_index;

	// 				while (true){

	// 					cg::coalesced_group searching_group = cg::coalesced_threads();

	// 					my_index = ext_bits.get_random_active_bit();

	// 					#if DEBUG_PRINTS
	// 					if (searching_group.thread_rank() == 0){
	// 						printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
	// 					}
	// 					#endif

	// 					//any threads still searching group together
	// 					//do an exclusive scan on the OR bits 

	// 					//if the exclusive OR result doesn't contain your bit you are free to modify!

	// 					//last thread knows the true state of the system, so broadcast changes.

						

	// 					uint64_t my_mask = (1ULL) << my_index;

	// 					//now scan across the masks
	// 					uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

	// 					//final thread needs to broadcast updates
	// 					if (searching_group.thread_rank() == searching_group.size()-1){

	// 						//doesn't matter as the scan only adds bits
	// 						//not to set the mask to all bits not taken
	// 						uint64_t final_mask = ~(scanned_mask | my_mask);

	// 						ext_bits.apply_mask(final_mask);

	// 					}

	// 					//everyone now has an updated final copy of ext bits?
	// 					ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


	// 					if (!(scanned_mask & my_mask)){

	// 						//I received an item!
	// 						//allocation has already been marked and index is set
	// 						//break to recoalesce for exit
	// 						break;



	// 					}


	// 				} //internal while loop

	// 				coalesced_threads.sync();

	// 				//TODO - take offset based on alloc size
	// 				//for now these are one byte allocs
	// 				allocation = (void *) (memmap[allocation_index_bit] + my_index);

	// 				//someone now has the minimum.
	// 				int my_fill = ext_bits.get_fill();

	// 				int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

	// 				int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

	// 				#if DEBUG_PRINTS
	// 				if (leader == coalesced_threads.thread_rank()){
	// 					printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
	// 				}
	// 				#endif
	// 				//printf("Leader is %d\n", leader, coalesced_threads.size());


	// 				if ((ext_bits.get_fill() > 0) && (leader == coalesced_threads.thread_rank())){

	// 					attach_buffer(memmap, ext_bits);

	// 				}

	// 				return true;




	// 			} //if active alloc


	// 		} //if bit set

			


	// 		//one extra inserted above this
	// 		//on failure reload local copy
	// 		local_copy = manager_bits.global_load_this();

	// 		} //current end of while loop?

	// 	return false;	

	// }

	




};


//some phase returns ~0 as the lock?
__device__ bool alloc_with_locks(uint64_t & allocation, offset_alloc_bitarr * manager, offset_storage_bitmap * block_storage){

	uint64_t remainder = 0ULL;

	__shared__ warp_lock team_lock;

	while (true){

		cg::coalesced_group grouping = cg::coalesced_threads();

		bool ballot = false;

		if (grouping.thread_rank() == 0){	

			//one thread groups;

			ballot = team_lock.lock();

		}

		if (grouping.ballot(ballot)) break;

		//printf("Team stuck in lock?\n");

	}

	cg::coalesced_group in_lock = cg::coalesced_threads();

	__threadfence();

	// if (in_lock.thread_rank() == 0){
	// 	printf("Team of %d entering the lock\n", in_lock.size());
	// }


	//in lock is coalesced team;
	bool ballot = (block_storage->bit_malloc_v3(in_lock, allocation, remainder));




	#if SLAB_PRINT_DEBUG
	if (ballot && (allocation == ~0ULL)){
		printf("Bug in first malloc, remainder is %llu\n", remainder);
	}
	#endif


	//if 100% of requests are satisfied, we are all returning, so one thread needs to drop lock.
	if ( __popc(in_lock.ballot(ballot)) == in_lock.size()){

		if (in_lock.thread_rank() == 0){

			if (__popcll(remainder) > 0){
				block_storage->attach_buffer(allocation - (allocation % 64), remainder);
			}

			team_lock.unlock();
		}

	}

	if (ballot){
		return true;
	}



	cg::coalesced_group remaining = cg::coalesced_threads();

	// if (remaining.thread_rank() == 0){
	// 	printf("Team of size %d remaining\n", remaining.size());
	// }
	//everyone else now can access the main alloc
	//void * remainder_offset;
	//bool is_leader = false;

	//this can't partially fail - I think.
	//should only ever return teams of 64 or total bust
	bool bit_malloc_result = manager->bit_malloc_v2(remaining, allocation, remainder);



	if (bit_malloc_result){

		if (!manager->belongs_to_block(allocation)){
			printf("Primary Offset bug.\n");
		}

		//uint64_t debug_alloc_bitarr_offset = allocation/4096;

		// offset_alloc_bitarr * alt_bitarr = (offset_alloc_bitarr *) block_allocator->get_mem_from_offset(debug_alloc_bitarr_offset);

		// if (alt_bitarr != bitarr){
		// 	printf("Alt bitarr bug\n");
		// }

	}



	if (remaining.thread_rank() == 0){
	      
		  //only attempt to attach if not empty.
	      if (__popcll(remainder) > 0  && bit_malloc_result){
		      bool result = block_storage->attach_buffer(allocation - (allocation % 64), remainder);

		      #if SLAB_PRINT_DEBUG
		      if (!result){
		      	printf("Failed to attach - this is a bug\n");
		      }

		      #endif

	  		}
	      
	      team_lock.unlock();

	}

	__threadfence();


	return bit_malloc_result;


}


__global__ void init_storage(offset_storage_bitmap * storages, int num_storages){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_storages) return;

	storages[tid].init();

}

struct pinned_storage {

	offset_storage_bitmap * storages;



	static __host__ pinned_storage * generate_on_device(int device){

		pinned_storage * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(pinned_storage));

		offset_storage_bitmap * dev_storages;


		int num_storages = poggers::utils::get_num_streaming_multiprocessors(device);

		printf("Booting up %d storages, %llu bytes\n", num_storages, sizeof(offset_storage_bitmap)*num_storages);
		cudaMalloc((void **)&dev_storages, sizeof(offset_storage_bitmap)*num_storages);

		init_storage<<<(num_storages-1)/256+1,256>>>(dev_storages, num_storages);

		cudaDeviceSynchronize();

		host_storage->storages = dev_storages;

		pinned_storage * dev_ptr;

		cudaMalloc((void **)&dev_ptr, sizeof(pinned_storage));

		cudaMemcpy(dev_ptr, host_storage, sizeof(pinned_storage), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_storage);

		return dev_ptr;


	}

	//if you don't specify we go on device 0.
	static __host__ pinned_storage * generate_on_device(){

		return generate_on_device(0);
	}


	static __host__ void free_on_device(pinned_storage * dev_storage){

		pinned_storage * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(pinned_storage));

		cudaMemcpy(host_storage, dev_storage, sizeof(pinned_storage), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(dev_storage);

		cudaFree(host_storage->storages);

		cudaFreeHost(host_storage);

		return;


	}

	__device__ offset_storage_bitmap * get_pinned_storage(){

		return &storages[poggers::utils::get_smid()];

	}


};


// __device__ bool alloc_with_locks_no_storage(uint64_t & allocation, offset_alloc_bitarr * manager, offset_storage_bitmap * block_storage){

// 	uint64_t remainder = 0ULL;

// 	__shared__ warp_lock team_lock;

// 	while (true){

// 		cg::coalesced_group grouping = cg::coalesced_threads();

// 		bool ballot = false;

// 		if (grouping.thread_rank() == 0){	

// 			//one thread groups;

// 			ballot = team_lock.lock();

// 		}

// 		if (grouping.ballot(ballot)) break;

// 	}


// 	cg::coalesced_group remaining = cg::coalesced_threads();

// 	//everyone else now can access the main alloc
// 	//void * remainder_offset;
// 	//bool is_leader = false;

// 	//this can't partially fail - I think.
// 	//should only ever return teams of 64 or total bust
// 	bool bit_malloc_result = manager->bit_malloc_v2(remaining, allocation, remainder);



// 		if (remaining.thread_rank() == 0 ){
	      
// 		  //only attempt to attach if not empty.
// 	      if (__popcll(remainder) > 0){
// 		      //bool result = block_storage->attach_buffer(allocation - (allocation % 64), remainder);


// 	  		}
	      
// 	      team_lock.unlock();

// 	}

// 	__threadfence();


// 	return bit_malloc_result;


// }

//one of these guys is how many bytes?

//exactly
//2^14 bytes
//31*(8*66)+16

// template <uint64_t alloc_size>
// struct slab_retreiver {

// 	uint64_t_bitarr slabs;
// 	void * memory;
// 	alloc_bitarr bitmaps[31];



// 	__device__ void init(void * ext_memory){
// 		//set all but top bit
// 		slabs = BITMASK(31);

// 		memory = ext_memory;

// 	}


// 	__device__ alloc_bitarr * give_bitmap(){

// 		while (true){

// 			int index = slabs.get_random_active_bit();

// 			if (index == -1 ) return nullptr;

// 			if (slabs.unset_index(index) & SET_BIT_MASK(index)){

// 				bitmaps[index].init();
// 				bitmaps[index].attach_allocation(memory + 4096*alloc_size*index);

// 				return &bitmaps[index];

// 			}

// 		}





// 	}


// 	//return a bitmap that belongs to this allocator.
// 	__device__ bool free_bitmap(alloc_bitarr * bitmap){

// 		bitmap->manager_bits.global_load_this();
// 		assert(bitmap->manager_bits == (~0ULL));

// 		int index = ((uint64_t) bitmap - (uint64_t ) bitmaps)/sizeof(alloc_bitarr);

// 		return (slabs.set_index(index) | SET_BIT_MASK(index));

// 	}

// 	__device__ void * get_mem_ptr(){
// 		return memory;
// 	}



// };


//states
//11--alloced and ready to go
//01 - alloced
template <int num_backups>
struct smid_pinned_storage {


	static_assert(num_backups < 64);

	//one bit buffer?
	uint64_t_bitarr slab_markers;


	offset_alloc_bitarr * slab_ptrs[num_backups+1];


	__device__ void init(){

		slab_markers = 0ULL;


		for (int i = 0; i < num_backups+1; i++){

			slab_ptrs[i] = nullptr;

		}

	}



	//ISSUE
	//non primary may be passed in here and non be the primary... - thats bad mkay
	//to rectify need to detect non match and swap out?

	//reserve that you are exclusive manager - unset manager bit
	//todo - only swap with known value of primary - prevent unneccesary swap outs


	__device__ bool pivot_non_primary(int index, offset_alloc_bitarr * old_item){


		if (!(slab_markers.unset_index(index) & SET_BIT_MASK(index))){

			//claimed by someone else
			return false;

		}

	}

	__device__ int pivot_primary(offset_alloc_bitarr * old_primary){


		//printf("Thread %d entering pivot\n", threadIdx.x);

		if (!(slab_markers.unset_index(0) & SET_BIT_MASK(0))){
			return -1;
		}

		old_primary->mark_unpinned();

		if (!old_primary->atomic_check_unpinned()){
			printf("Internal pivot unpin bug\n");
		}

		//not happening - so the blocks must be full at some point...
		if (old_primary->is_full_atomic()){
			printf("behavior bug in pivot primary\n");
		}

		__threadfence();

		while (true){

			int index = slab_markers.get_random_active_bit_nonzero();

			if (index == -1){
				return -1;
			}

			if (slab_markers.unset_index(index) & SET_BIT_MASK(index)){

				//legal and very cool
				//other threads must check that they do not receive a nullptr in this rather unstable state.
				uint64_t old = atomicExch((unsigned long long int *) &slab_ptrs[index], 0ULL);

				if (atomicCAS((unsigned long long int *)&slab_ptrs[0], (unsigned long long int) old_primary, (unsigned long long int) old) != (unsigned long long int) old_primary){

					//printf("This is the fail case\n");

					slab_markers.set_index(0);
					attach_new_buffer(index, (offset_alloc_bitarr *) old);
					return -1;

				}

				slab_markers.set_index(0);

				//printf("Index %d successfully swapped to primary\n", index);
				return index;


			} else {

				//someone else has grabbed an index for alloc - this should be impossible?
				printf("This may be undefined behavior\n");
			}


		}

	}


	//potential bug - pivot allows for multiple to succeed 
	//accidentally swapping valud blocks
	//adding this bit check appears to solve - check extensively	
	__device__ offset_alloc_bitarr * get_primary(){

		offset_alloc_bitarr * ret = slab_ptrs[0];

		// if (ret == nullptr || !(slab_markers.bits & SET_BIT_MASK(0))){
		// 	return get_non_primary();
		// }


		if ((uint64_t) ret == 0x1){

			printf("Bug inside get primary\n");

		}

		return ret;

		//int valid_index = poggers::utils::get_smid(); 



	}

	__device__ offset_alloc_bitarr * get_non_primary(){

		//multiple threads in the same warp should maintain coalescing.

		int index = slab_markers.get_random_active_bit_warp();

		if (index == -1) return nullptr;

		// if ( (uint64_t) slab_ptrs[index] == 0x1){
		// 	printf("Bug in get_non_primary\n");
		// }

		return slab_ptrs[index];
	}


	// __device__ bool attach_new_buffer(int index, offset_alloc_bitarr * new_buffer){

	// 	if (slab_markers.set_index(index) & SET_BIT_MASK(index)){

	// 		printf("Error attaching: index %d already set\n", index);
	// 		return false;

	// 	} else {


	// 		uint64_t old = atomicExch((unsigned long long int *)&slab_ptrs[index], (unsigned long long int) new_buffer);

	// 		if (old != 0ULL){

	// 			printf("%d Exchanged for an already set buffer: %llx exchanged\n", index, old);

	// 			//weird state but I think this is technically a success
	// 			return true;

	// 		}

	// 		//printf("Buffer attached to index %d\n", index);
	// 		return true;


	// 	}

	// }


	__device__ bool attach_new_buffer(int index, offset_alloc_bitarr * new_buffer){

		uint64_t old = atomicCAS((unsigned long long int *)&slab_ptrs[index], 0ULL, (unsigned long long int) new_buffer);

		if (old != 0ULL){

			printf("%d Exchanged for an already set buffer: %llx exchanged\n", index, old);

			return false;

		}

		if (slab_markers.set_index(index) & SET_BIT_MASK(index)){

			printf("Error attaching: index %d already set\n", index);

			return false;
		}

		return true;

	}


	template<typename block_allocator, typename memory_allocator>
	__device__ void init_with_allocators(block_allocator * balloc, memory_allocator * memalloc){

		//boot myself to clear memory
		init();


		for (int i = 0; i < num_backups+1; i++){

			offset_alloc_bitarr * slab = (offset_alloc_bitarr *) balloc->malloc();

			if (slab == nullptr){
				printf("Failed to load slab from allocator\n");
				return;
			}

			uint64_t offset = memalloc->get_offset();

			if (offset == memory_allocator::fail()){
				balloc->free(slab);
				printf("Fail to claim memory for slab\n");

			}

			//don't forget to actually boot memory lol
			slab->init();

			slab->attach_allocation(offset);

			slab->mark_pinned();

			attach_new_buffer(i, slab);

			
		}


	}		

		template<typename block_allocator>
	__device__ void init_with_allocators_memory(block_allocator * balloc, uint64_t ext_offset){

		//boot myself to clear memory
		init();


		for (int i = 0; i < num_backups+1; i++){

			uint64_t slab_offset = balloc->get_offset();

			offset_alloc_bitarr * slab = (offset_alloc_bitarr *) balloc->get_mem_from_offset(slab_offset);

			if (slab == nullptr){
				printf("Failed to load slab from allocator\n");
				return;
			}

			uint64_t offset = slab_offset*ext_offset; 

			// if (offset == memory_allocator::fail()){
			// 	balloc->free(slab);
			// 	printf("Fail to claim memory for slab\n");

			// }

			//don't forget to actually boot memory lol
			slab->init();

			slab->attach_allocation(offset);

			slab->mark_pinned();

			attach_new_buffer(i, slab);

			
		}


	}


};



template <int num_blocks, typename block_allocator, typename memory_allocator>
__global__ void smid_pinned_block_init_storage(block_allocator * balloc, memory_allocator * memalloc, smid_pinned_storage<num_blocks> * storages, int num_storages){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_storages) return;

	storages[tid].init_with_allocators(balloc, memalloc);

}

template <int num_blocks, typename block_allocator>
__global__ void smid_pinned_block_init_storage_char(block_allocator * balloc, uint64_t offset, smid_pinned_storage<num_blocks> * storages, int num_storages){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_storages) return;

	storages[tid].init_with_allocators_memory(balloc, offset);

}

//storage pinned buffer

template <int num_blocks>
struct smid_pinned_container {


	using my_type = smid_pinned_container<num_blocks>;

	using pinned_type = smid_pinned_storage<num_blocks>;

	pinned_type * storages;


	template <typename block_allocator, typename memory_allocator>
	static __host__ my_type * generate_on_device(int device, block_allocator * balloc, memory_allocator * memalloc){

		my_type * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(my_type));

		pinned_type * dev_storages;


		int num_storages = poggers::utils::get_num_streaming_multiprocessors(device);

		printf("Booting up %d storages, %llu bytes\n", num_storages, sizeof(pinned_type)*num_storages);
		cudaMalloc((void **)&dev_storages, sizeof(pinned_type)*num_storages);

		smid_pinned_block_init_storage<num_blocks, block_allocator, memory_allocator><<<(num_storages-1)/256+1,256>>>(balloc, memalloc, dev_storages, num_storages);

		cudaDeviceSynchronize();

		host_storage->storages = dev_storages;

		my_type * dev_ptr;

		cudaMalloc((void **)&dev_ptr, sizeof(my_type));

		cudaMemcpy(dev_ptr, host_storage, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_storage);

		return dev_ptr;


	}

	//if you don't specify we go on device 0.
	template <typename block_allocator, typename memory_allocator>
	static __host__ my_type * generate_on_device(block_allocator * balloc, memory_allocator * memalloc){

		return my_type::generate_on_device(0, balloc, memalloc);
	}


	template <typename block_allocator>
	static __host__ my_type * generate_on_device(int device, block_allocator * balloc, uint64_t ext_offset){

		my_type * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(my_type));

		pinned_type * dev_storages;


		int num_storages = poggers::utils::get_num_streaming_multiprocessors(device);

		printf("Booting up %d storages, %llu bytes\n", num_storages, sizeof(pinned_type)*num_storages);
		cudaMalloc((void **)&dev_storages, sizeof(pinned_type)*num_storages);

		smid_pinned_block_init_storage_char<num_blocks, block_allocator><<<(num_storages-1)/256+1,256>>>(balloc, ext_offset, dev_storages, num_storages);

		cudaDeviceSynchronize();

		host_storage->storages = dev_storages;

		my_type * dev_ptr;

		cudaMalloc((void **)&dev_ptr, sizeof(my_type));

		cudaMemcpy(dev_ptr, host_storage, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_storage);

		return dev_ptr;


	}

	//if you don't specify we go on device 0.
	template <typename block_allocator>
	static __host__ my_type * generate_on_device(block_allocator * balloc, uint64_t ext_offset){

		return my_type::generate_on_device(0, balloc, ext_offset);
	}


	static __host__ void free_on_device(my_type * dev_storage){

		my_type * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(my_type));

		cudaMemcpy(host_storage, dev_storage, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(dev_storage);

		cudaFree(host_storage->storages);

		cudaFreeHost(host_storage);

		return;


	}

	__device__ pinned_type * get_pinned_storage(){

		return &storages[poggers::utils::get_smid()];

	}

	static __host__ my_type * generate_on_device(){

		my_type * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(my_type));

		pinned_type * dev_storages;

		int num_storages = poggers::utils::get_num_streaming_multiprocessors(0);

		printf("Booting up %d storages, %llu bytes\n", num_storages, sizeof(pinned_type)*num_storages);
		cudaMalloc((void **)&dev_storages, sizeof(pinned_type)*num_storages);

		//smid_pinned_block_init_storage_char<num_blocks, block_allocator><<<(num_storages-1)/256+1,256>>>(balloc, ext_offset, dev_storages, num_storages);

		cudaDeviceSynchronize();

		host_storage->storages = dev_storages;

		my_type * dev_ptr;

		cudaMalloc((void **)&dev_ptr, sizeof(my_type));

		cudaMemcpy(dev_ptr, host_storage, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_storage);

		return dev_ptr;



	}

};






//was just curious - this verifies that the host does not boot items on kernel start
//so __shared just get initialized to 0

// struct kernel_init_test {

// 	__device__ kernel_init_test(){
// 		printf("Booting up! controlled by %llu\n", threadIdx.x+blockIdx.x*blockDim.x);
// 	}

// 	__device__ ~kernel_init_test(){
// 		printf("Shutting down! controlled by %llu\n", threadIdx.x+blockIdx.x*blockDim.x);
// 	}




// };



}

}


#endif //GPU_BLOCK_