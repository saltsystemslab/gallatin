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


namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 




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



struct alloc_bitarr{

	void * memmap;
	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];

	__device__ void init(){

		manager_bits.bits = ~(0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = ~(0ULL);
		}
		//at some point work on this
		memmap = nullptr;

	}


	__device__ void attach_allocation(void * ext_alloc){

		memmap = ext_alloc;

	}

	//request one item for this thread
	__device__ bool bit_malloc(void * & allocation, uint64_t & remainder, void * & remainder_offset, bool & is_leader){


		//group
		//

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();


		#if DEBUG_PRINTS
		cg::coalesced_group active_threads = cg::coalesced_threads();

		if (active_threads.thread_rank() == 0){
			printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
		}
		#endif
		

		while(local_copy.get_fill() != 0){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			int allocation_index_bit = 0;

			if (active_threads.thread_rank() == 0){

				//allocation_index_bit = local_copy.get_first_active_bit();

				allocation_index_bit = local_copy.get_random_active_bit();

			}
			
			allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

			uint64_t_bitarr ext_bits;

			bool ballot_bit_set = false;

			if (active_threads.thread_rank() == 0){


				if (manager_bits.unset_bit_atomic(allocation_index_bit)){


					ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

					ballot_bit_set = true;



				}


			}

			//at this point, ballot_bit_set and ext_bits are set in thread 0
			//so we ballot on if we can leave the loop

			if (active_threads.ballot(ballot_bit_set)){


				 
				ext_bits = active_threads.shfl(ext_bits, 0);

				#if DEBUG_PRINTS
				if (active_threads.thread_rank() == 0){
					printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
				}
				#endif


				if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

					//next step: gather threads
					cg::coalesced_group coalesced_threads = cg::coalesced_threads();

					#if DEBUG_PRINTS
					if (coalesced_threads.thread_rank() == 0){
						printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
					}
					#endif


					//how to sync outputs?
					//everyone should pick a random lane?

					//how to coalesce after lanes are picked


					//options
					//1) grab an allocation of the first n and try to  
					//2) select the first n bits ahead of time.

					//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

					//int my_bits = bits_before_index(active_threads.thread_rank());

					// bool ballot = (bits_needeed == my_bits);

					// int result = coalesced_threads.ballot(ballot);

					
					int my_index;

					while (true){

						cg::coalesced_group searching_group = cg::coalesced_threads();

						my_index = ext_bits.get_random_active_bit();

						#if DEBUG_PRINTS
						if (searching_group.thread_rank() == 0){
							printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
						}
						#endif

						//any threads still searching group together
						//do an exclusive scan on the OR bits 

						//if the exclusive OR result doesn't contain your bit you are free to modify!

						//last thread knows the true state of the system, so broadcast changes.

						

						uint64_t my_mask = (1ULL) << my_index;

						//now scan across the masks
						uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

						//final thread needs to broadcast updates
						if (searching_group.thread_rank() == searching_group.size()-1){

							//doesn't matter as the scan only adds bits
							//not to set the mask to all bits not taken
							uint64_t final_mask = ~(scanned_mask | my_mask);

							ext_bits.apply_mask(final_mask);

						}

						//everyone now has an updated final copy of ext bits?
						ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


						if (!(scanned_mask & my_mask)){

							//I received an item!
							//allocation has already been marked and index is set
							//break to recoalesce for exit
							break;



						}


					} //internal while loop

					coalesced_threads.sync();

					//TODO - take offset based on alloc size
					//for now these are one byte allocs
					allocation = (void *) (memmap + my_index + 64*allocation_index_bit);

					//someone now has the minimum.
					int my_fill = ext_bits.get_fill();

					int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

					int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

					#if DEBUG_PRINTS
					if (leader == coalesced_threads.thread_rank()){
						printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
					}
					#endif
					//printf("Leader is %d\n", leader, coalesced_threads.size());

					if ((leader == coalesced_threads.thread_rank())){

						is_leader = true;
						remainder = ext_bits;

						remainder_offset = memmap + 64*allocation_index_bit;

					} else {

						is_leader = false;
						remainder = 0;
						remainder_offset = nullptr;

					}


					return true;




				} //if active alloc


			} //if bit set

			


			//one extra inserted above this
			//on failure reload local copy
			local_copy = manager_bits.global_load_this();

			} //current end of while loop?

		return false;	

	}


	//TODO: templatize over size in bytes
	__device__ bool bit_free(void * allocation, uint64_t size_in_bytes){


		int my_offset = ((uint64_t) allocation - (uint64_t) memmap)/size_in_bytes;

		int upper_bit = my_offset/64;

		int lower_bit = my_offset % 64; 


		//collate thread teams together
		while (true){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			//only threads that match with the leader may progress.

			int team_upper_bit = active_threads.shfl(upper_bit, 0);

			if (team_upper_bit == upper_bit) break;

			

		}

		//starting team now shares the same upper bit
		cg::coalesced_group starting_team = cg::coalesced_threads();


		uint64_t my_mask = (1ULL << lower_bit);

		uint64_t scanned_mask = cg::inclusive_scan(starting_team, my_mask, cg::bit_or<uint64_t>());

		if (starting_team.thread_rank() == starting_team.size()-1){

			

			if (alloc_bits[upper_bit].set_OR_mask(scanned_mask) | scanned_mask == (~0ULL)){

				manager_bits.set_bit_atomic(upper_bit);

				return manager_bits | SET_BIT_MASK(upper_bit);

			}

		}


		return false;


	}

	




};



//Correctness precondition
//0000000000000000 is empty key
//if you create it you *will* destroy it
//so other threads don't touch blocks that show themselves as 0ULL
//This allows it to act as the intermediate state of blocks
//and allows the remove pipeline to be identical to above ^
//as we first remove and then re-add if there are leftovers.
struct storage_bitmap{


	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];
	void * memmap[64];


	__device__ void init(){

		manager_bits.bits = (0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = (0ULL);
			memmap[i] = nullptr;
		}


		

	}


	__device__ bool attach_buffer(void * ext_buffer, uint64_t ext_bits){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();

		while (local_copy.get_fill() != 64){

			local_copy.invert();

			#if DEBUG_PRINTS
			printf("Copy: %llx\n", local_copy);
			#endif


				//allocation_index_bit = local_copy.get_first_active_bit();

			int allocation_index_bit = local_copy.get_random_active_bit();


			#if DEBUG_PRINTS
			printf("Bit chosen is %d / %llx, %llx\n", allocation_index_bit, manager_bits, alloc_bits[allocation_index_bit]);
			#endif

			if (alloc_bits[allocation_index_bit].set_bits(ext_bits)){



				if (manager_bits.set_bit_atomic(allocation_index_bit)){

					#if DEBUG_PRINTS
					printf("Manager bit set!\n");
					#endif
				

					return true;

				} else {
					//if you swap out you *must* succeed
					printf("Failure attaching buffer\n");
					assert(1==0);
				}


			}


			local_copy = manager_bits.global_load_this();


		}


		return false;


	}


	__device__ bool bit_malloc(void * & allocation){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();

		#if DEBUG_PRINTS
		if (active_threads.thread_rank() == 0){
			printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
		}
		#endif
		

		while(local_copy.get_fill() != 0ULL){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			int allocation_index_bit = 0;

			//does removing this gate affect performance?

			if (active_threads.thread_rank() == 0){

				//allocation_index_bit = local_copy.get_first_active_bit();

				allocation_index_bit = local_copy.get_random_active_bit();

			}
			
			allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

			uint64_t_bitarr ext_bits;

			bool ballot_bit_set = false;

			if (active_threads.thread_rank() == 0){


				if (manager_bits.unset_bit_atomic(allocation_index_bit)){


					ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

					ballot_bit_set = true;



				}


			}

			//at this point, ballot_bit_set and ext_bits are set in thread 0
			//so we ballot on if we can leave the loop

			if (active_threads.ballot(ballot_bit_set)){


				 
				ext_bits = active_threads.shfl(ext_bits, 0);

				#if DEBUG_PRINTS
				if (active_threads.thread_rank() == 0){
					printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
				}
				#endif


				if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

					//next step: gather threads
					cg::coalesced_group coalesced_threads = cg::coalesced_threads();

					#if DEBUG_PRINTS
					if (coalesced_threads.thread_rank() == 0){
						printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
					}
					#endif

					//how to sync outputs?
					//everyone should pick a random lane?

					//how to coalesce after lanes are picked


					//options
					//1) grab an allocation of the first n and try to  
					//2) select the first n bits ahead of time.

					//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

					//int my_bits = bits_before_index(active_threads.thread_rank());

					// bool ballot = (bits_needeed == my_bits);

					// int result = coalesced_threads.ballot(ballot);

					
					int my_index;

					while (true){

						cg::coalesced_group searching_group = cg::coalesced_threads();

						my_index = ext_bits.get_random_active_bit();

						#if DEBUG_PRINTS
						if (searching_group.thread_rank() == 0){
							printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
						}
						#endif

						//any threads still searching group together
						//do an exclusive scan on the OR bits 

						//if the exclusive OR result doesn't contain your bit you are free to modify!

						//last thread knows the true state of the system, so broadcast changes.

						

						uint64_t my_mask = (1ULL) << my_index;

						//now scan across the masks
						uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

						//final thread needs to broadcast updates
						if (searching_group.thread_rank() == searching_group.size()-1){

							//doesn't matter as the scan only adds bits
							//not to set the mask to all bits not taken
							uint64_t final_mask = ~(scanned_mask | my_mask);

							ext_bits.apply_mask(final_mask);

						}

						//everyone now has an updated final copy of ext bits?
						ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


						if (!(scanned_mask & my_mask)){

							//I received an item!
							//allocation has already been marked and index is set
							//break to recoalesce for exit
							break;



						}


					} //internal while loop

					coalesced_threads.sync();

					//TODO - take offset based on alloc size
					//for now these are one byte allocs
					allocation = (void *) (memmap[allocation_index_bit] + my_index);

					//someone now has the minimum.
					int my_fill = ext_bits.get_fill();

					int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

					int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

					#if DEBUG_PRINTS
					if (leader == coalesced_threads.thread_rank()){
						printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
					}
					#endif
					//printf("Leader is %d\n", leader, coalesced_threads.size());


					if ((ext_bits.get_fill() > 0) && (leader == coalesced_threads.thread_rank())){

						attach_buffer(memmap, ext_bits);

					}

					return true;




				} //if active alloc


			} //if bit set

			


			//one extra inserted above this
			//on failure reload local copy
			local_copy = manager_bits.global_load_this();

			} //current end of while loop?

		return false;	

	}

	




};


__device__ bool alloc_with_locks(void *& allocation, alloc_bitarr * manager, storage_bitmap * block_storage){

	__shared__ warp_lock team_lock;

	while (true){

		cg::coalesced_group grouping = cg::coalesced_threads();

		bool ballot = false;

		if (grouping.thread_rank() == 0){	

			//one thread groups;

			ballot = team_lock.lock();

		}

		if (grouping.ballot(ballot)) break;

	}

	cg::coalesced_group in_lock = cg::coalesced_threads();
	//team has locked
	bool ballot = false;

	if (block_storage->bit_malloc(allocation)){

		ballot = true;
	}

	//if 100% of requests are satisfied, we are all returning, so one thread needs to drop lock.
	if ( __popc(in_lock.ballot(ballot)) == in_lock.size()){

		if (in_lock.thread_rank() == 0){
			team_lock.unlock();
		}

	}

	if (ballot){
		return true;
	}

	//everyone else now can access the main alloc

	uint64_t remainder;
	void * remainder_offset;
	bool is_leader = false;

	bool bit_malloc_result = manager->bit_malloc(allocation, remainder, remainder_offset, is_leader);

	if (is_leader){
	      
	      bool result = block_storage->attach_buffer(remainder_offset, remainder);
	      
	      team_lock.unlock();
	}

	return bit_malloc_result;


}


//one of these guys is how many bytes?

//exactly
//2^14 bytes
//31*(8*66)+16

template <uint64_t alloc_size>
struct slab_retreiver {

	uint64_t_bitarr slabs;
	void * memory;
	alloc_bitarr bitmaps[31];



	__device__ void init(void * ext_memory){
		//set all but top bit
		slabs = BITMASK(31);

		memory = ext_memory;

	}


	__device__ alloc_bitarr * give_bitmap(){

		while (true){

			int index = slabs.get_random_active_bit();

			if (index == -1 ) return nullptr;

			if (slabs.unset_index(index) & SET_BIT_MASK(index)){

				bitmaps[index].init();
				bitmaps[index].attach_allocation(memory + 4096*alloc_size*index);

				return &bitmaps[index];

			}

		}





	}


	//return a bitmap that belongs to this allocator.
	__device__ bool free_bitmap(alloc_bitarr * bitmap){

		bitmap->manager_bits.global_load_this();
		assert(bitmap->manager_bits == (~0ULL));

		int index = ((uint64_t) bitmap - (uint64_t ) bitmaps)/sizeof(alloc_bitarr);

		return (slabs.set_index(index) | SET_BIT_MASK(index));

	}

	__device__ void * get_mem_ptr(){
		return memory;
	}



};


template <uint64_t alloc_size>
struct slab_storage {


	uint64_t_bitarr slab_markers;

	alloc_bitarr * slab_ptrs[32];


	__device__ void init_random_index(alloc_bitarr * slab){


		while (true){


		slab_markers.global_load_this();

		//only need to replace 00
		//set to 10 to take ownership
		//then convert to 11 once threadfence ++ set.
		int index =  shrink_index(slab_markers.get_random_unset_bit_full());

		//int index = slab_markers.get_random_unset_bit();

		if (index == -1){

			printf("Weird bug in slab storage, too many allocations requested.\n");
			return;

		}

		if (!(slab_markers.set_control_bit_atomic(index) & SET_SECOND_BIT(index))){

			slab_ptrs[index] = slab;
			__threadfence();
			slab_markers.set_lock_bit_atomic(index);

			return;

		}


		}


	}

	__device__ void init_claimed_index(alloc_bitarr * slab, int index){


		slab_markers.global_load_this();

		//only need to replace 00
		//set to 10 to take ownership
		//then convert to 11 once threadfence ++ set.
		if (!(slab_markers.set_control_bit_atomic(index) & SET_SECOND_BIT(index))){

			slab_ptrs[index] = slab;
			__threadfence();
			slab_markers.set_lock_bit_atomic(index);

			printf("Thread %llu finished with index %d\n", threadIdx.x+blockIdx.x*blockDim.x, index);

			return;

		} else {

			printf("Bug, can't reset claimed index\n");
			assert(1==0);
		}




	}

	//This will be the source of a bug if you are too fast
	__device__ bool claim_index(int index){

		uint64_t old = slab_markers.unset_both_atomic(index);


		if (__popcll(old & READ_BOTH(index)) == 2){


			printf("Thread %llu claimed index %d\n", threadIdx.x+blockIdx.x*blockDim.x, index);

			return true;

		}

		if (__popcll(old & READ_BOTH(index)) == 0) return false;

		slab_markers.reset_both_atomic(old, index);

		return false;
	}

	__device__ int get_random_active_index(){


		return shrink_index(slab_markers.get_random_active_bit_full());

	}

	//return nullptr on malloc failure inside of local block
	//this is a signal to the allocator to go ahead and request a new block
	//this includes the malloc from the local mapping, so that must be passed in
	//this version utilizes the warp locks as well
	__device__ void * malloc(storage_bitmap * local_mapping, int index){


		
		void * my_allocation;

		if (alloc_with_locks(my_allocation, slab_ptrs[index], local_mapping)){

			return my_allocation;
		}

		return nullptr;

	}




};



}

}


#endif //GPU_BLOCK_