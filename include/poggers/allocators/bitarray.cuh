#ifndef BITARRAY
#define BITARRAY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include "stdio.h"
#include "assert.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;



#define BITMASK_CUTOFF 10

//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 


template <size_t bytes_per_item = 2>
struct bitarr_grouped {

	void * ptr_to_memory;
	uint64_t lead_bits;
	uint64_t bits[64];

	__device__ void init(){


		// lead_bits = ~0ULL;

		// for (int i =0; i < 64; i++){
		// 	bits[i] = ~0ULL;
		// }
		//I suspect this is faster but need to benchmark init.
		memset(this, 255, sizeof(bitarr_grouped));

		//unset the uppermost bit as it points to nothing.
		ptr_to_memory = nullptr;
	}



	//given an old allocation determine where I fit
	//guarantee that if you make it this far you will fit.
	__device__ int determine_my_alloc(uint64_t old_alloc){

		int start = __ffsll(old_alloc);

		while (true){


			cg::coalesced_group active_threads = cg::coalesced_threads();

			int my_start = start + active_threads.thread_rank();

			bool my_bit = old_alloc | (1ULL << my_start);

			if (my_bit) {

				return my_start;

			} else {

				old_alloc = old_alloc & ((~0ULL) << (start + active_threads.size()));

			}


		}
		


	}

	//should I pass a handle to the active group?
	__device__ int determine_my_alloc_relative(uint64_t * &address, uint64_t &metadata){


		while (true){

			//we can jump large unsets?
			int start = __ffsll(metadata)-1;

			cg::coalesced_group active_threads = cg::coalesced_threads();

			int my_start = start + active_threads.thread_rank();

			bool my_bit = metadata | (1ULL << my_start);

			//worst case you all got unset.
			metadata = metadata & ((~0ULL) << (start + active_threads.size()));

			if (my_bit) {

				//last thread guaranteed to have accurate metadata.
				return my_start;

			}

			#if DEBUG_PRINTS
			printf("Stalling in determine alloc?\n");
			#endif

		}


	}

	__device__ uint64_t check_attachment(){


		return poggers::utils::ldca(&lead_bits);

	}

	//uint64_t ** my_address, uint64_t * old_alloc
	//return a unique address for this item!
	__device__ uint64_t malloc(uint64_t this_id){

		

		uint64_t my_lead_bits = poggers::utils::ldca(&lead_bits);




		while (my_lead_bits){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			const auto leader = __ffsll(my_lead_bits)-1;

			//uint64_t working_bits = poggers::utils::ldca(bits+leader);
			uint64_t old_result;



			if (leader % active_threads.size() == active_threads.thread_rank()){

				//forcibly unset all;


				//this is buggy
				old_result = atomicExch((unsigned long long int *)(bits + leader), 0ULL);



				//and set lead to 0
				atomicAnd((unsigned long long int *)&lead_bits, ~(1ULL << leader));

			}

			//return my_lead_bits;

			old_result = active_threads.shfl(old_result, leader % active_threads.size());

			int allocations_left = __popcll(old_result);

			if (allocations_left > active_threads.thread_rank()){

				//I get an allocation!
				//old_alloc[0] = old_result;
				cg::coalesced_group mask_active_group = cg::coalesced_threads();

				return determine_my_alloc(old_result) + leader*64 + this_id;

				//here my_offset = determine_my_alloc(old_result);
				//and reconstruct new mask?

				} else {
					
					my_lead_bits ^= 1ULL << leader;
			}

			//printf("Stalling in malloc?\n");




			}

			//printf("Alloc failed\n");

			return 0;

		}



		//metadata malloc - this is called for all threads and controls some important functionality
		//this needs to return multiple values so the control 
		//should return - my malloced address
		// a bool suggesting that this block should be preempted
		// the address and uint64_t of any preallocations that I requested.
		__device__ cg::coalesced_group metadata_malloc(void * &my_malloc, bool& should_preempt, uint64_t * &address, uint64_t &remaining_metadata){

		uint64_t my_lead_bits = poggers::utils::ldca(&lead_bits);

		//if we detect that this bitarray is mostly allocated,
		//pop off a request to have it moved away.
		//should this be lower?
		if (__popcll(my_lead_bits) < BITMASK_CUTOFF){

			should_preempt = true;

		}


		while (my_lead_bits){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			const auto leader = __ffsll(my_lead_bits)-1;

			//uint64_t working_bits = poggers::utils::ldca(bits+leader);
			uint64_t old_result;



			if (leader % active_threads.size() == active_threads.thread_rank()){

				//forcibly unset all, and take ownership of any remaining.
				old_result = atomicExch((unsigned long long int *)(bits + leader), 0ULL);



				//and set lead to 0
				atomicAnd((unsigned long long int *)&lead_bits, ~(1ULL << leader));

			}

			//return my_lead_bits;

			old_result = active_threads.shfl(old_result, leader % active_threads.size());

			int allocations_left = __popcll(old_result);

			if (allocations_left > active_threads.thread_rank()){

				//I get an allocation!
				//old_alloc[0] = old_result;
				cg::coalesced_group mask_active_group = cg::coalesced_threads();

				//
				if (mask_active_group.size() != active_threads.size()){
					printf("Sync error here! %d, %d != %d\n", allocations_left, mask_active_group.size(), active_threads.size());
				}

				address = bits+leader;
				remaining_metadata = old_result;

				//there should be a scaling factor here
				my_malloc = (void *) ( (uint64_t) ptr_to_memory + bytes_per_item*(determine_my_alloc_relative(address, remaining_metadata) + leader));

				//the last true metadata will have the least bits set
				//no unset bits are reset
				//so lesser will reduce to the true lead.

				uint64_t new_metadata = cg::reduce(mask_active_group, remaining_metadata, cg::less<uint64_t>());


				assert(__popcll(new_metadata) <= __popcll(remaining_metadata));
				//mask_active_group.sync();

				return mask_active_group;
				//here my_offset = determine_my_alloc(old_result);
				//and reconstruct new mask?

				} else {
					
					my_lead_bits ^= 1ULL << leader;
				}


				#if DEBUG_PRINTS
				printf("Stalling in metadata malloc?\n");
				#endif



			}

			printf("Alloc failed\n");

			my_malloc = nullptr;
			//likely a team of just me?
			return cg::coalesced_threads();
		}







};


//parallel queue as a leftover bitmap
//set bits to 1 if available.

template <typename bitmap_type> 
__global__ void init_bitmaps(bitmap_type * bitmaps, uint64_t num_bitmaps){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid >= num_bitmaps) return;

	bitmaps[tid].init();

}

template <size_t bytes_per_item>
struct storage_bitmap {

	uint64_t lead_bits;
	uint64_t bits[64];
	void * addresses[64];

	using my_type = storage_bitmap<bytes_per_item>;


	//unlike grouped bitmaps, these are initialized to 0;
	//they get swapped to 1 as new requests come in. 
	__device__ void init(){

		lead_bits = 0ULL;
		memset(bits, 0, 64*sizeof(uint64_t));
		memset(addresses, 0, 64*sizeof(void *));

	}


	static __host__ my_type * generate_buffers(){

	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maximum_sm = prop.multiProcessorCount;

    printf("Booting with %d SMs.\n", maximum_sm);

    my_type * bitmaps;

    cudaMalloc((void **)&bitmaps, sizeof(my_type)*maximum_sm);

    init_bitmaps<my_type><<<(maximum_sm-1)/512+1, 512>>>(bitmaps, maximum_sm);

    return bitmaps;

	}

	static __host__ my_type * generate_buffers_blocks(uint64_t num_blocks){

	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    prop.multiProcessorCount;

    int maximum_sm = num_blocks;

    my_type * bitmaps;

    cudaMalloc((void **)&bitmaps, sizeof(my_type)*maximum_sm);

    init_bitmaps<my_type><<<(maximum_sm-1)/512+1, 512>>>(bitmaps, maximum_sm);

    return bitmaps;

	}

	__device__ static my_type * get_my_bitmap(my_type * bitmaps){

		int my_bitmap_id = poggers::utils::get_smid();

		//optional prefetch?
		poggers::utils::prefetch_l1(&bitmaps[my_bitmap_id]);		

		return &bitmaps[my_bitmap_id];

	}

	__device__ bool attach_buffer(void * my_buffer, uint64_t my_allocations){

		//find all unset values
		uint64_t my_lead_bits = ~poggers::utils::ldca(&lead_bits);
		uint64_t orig_lead_bits = my_lead_bits;

		//printf("my_lead_bits %llx\n", my_lead_bits);


		//there is a stall in this loop.

		while (my_lead_bits){


			//SETUP - this isn't working because the other system does not properly drain the main ihput. 
			//printf("Stalling %d\n", __popcll(my_lead_bits));

			

			
			const auto leader = __ffsll(my_lead_bits)-1;	

			uint64_t old = atomicCAS((unsigned long long int *)(bits+leader), 0ULL, (unsigned long long int) my_allocations);

			// if (printing){
			// 	printing = false;
			// 	printf("old %llx\n", old);
			// }
			

			if (old  != 0ULL){
				//unsuccessful! retry;
				//my_lead_bits ^= 1ULL << leader;
				//try a reload?
				my_lead_bits = ~poggers::utils::ldca(&lead_bits);
			} else {
				//success! swap out buffer and set flag
				//so someone else can try;
				atomicExch((unsigned long long int *)(addresses +leader), (unsigned long long int) my_buffer);

				//printf("bit to change: %llx\n", (1ULL << leader));
				atomicOr((unsigned long long int *)&lead_bits, (1ULL << leader));

				__threadfence();

				//this change is not broadcast? update to LDCA fixed it!

				//printf("NEW LEAD: %llx\n", ~poggers::utils::ldca(&lead_bits));

				return true;


			}

			#if DEBUG_PRINTS
			printf("Stalling in attach phase?\n");
			#endif

		}

		printf("Failure to add new buffer. Shoudn't be possible? %llx orig lead\n", orig_lead_bits);
		return false;

	}


	__device__ uint64_t check_attachment(){


		return poggers::utils::ldca(&lead_bits);

	}

	__device__ void * malloc_from_existing(){

		__threadfence();

		uint64_t my_lead_bits = poggers::utils::ldca(&lead_bits);

		// if (my_lead_bits != 0ULL){

		// 	printf("Interesting\n");
		// }

		//if we detect that this bitarray is mostly allocated,
		//pop off a request to have it moved away.
		//should this be lower?

		//printf("My lead bits in malloc existing: %llx\n", my_lead_bits);


		while (my_lead_bits){

			

			const auto leader = __ffsll(my_lead_bits)-1;

			uint64_t old_result = poggers::utils::ldca(&bits[leader]);

			//atomic swap
			while (old_result){

				//moved active_threads into the lower partition so that it is refreshed by the loop;
				cg::coalesced_group active_threads = cg::coalesced_threads();

				int start = __ffsll(old_result)-1;

				uint64_t mask = (~0ULL << (start + active_threads.size()));

				

				if (active_threads.thread_rank() == 0){
					old_result = atomicAnd((unsigned long long int *)&bits[leader], mask);
				}

				old_result = active_threads.shfl(old_result, 0);


				printf("Mask %llx | current value: %llx\n", mask, old_result);

				if (old_result | (1ULL << (start + active_threads.thread_rank()))){

					//any mallocs that finish off do swap.

					if ( (old_result & mask) == 0ULL && active_threads.thread_rank() == 0){

						printf("Unsetting %d\n", leader);
						atomicAnd((unsigned long long int *)&lead_bits, ~(1ULL << leader));
					}

					return (void *) (( (uint64_t) addresses[leader]) + bytes_per_item*(start+active_threads.thread_rank()));

				}

				//everyone else can drop;
				old_result = old_result & mask;


			}



			my_lead_bits ^= (1ULL << leader);


		}

		//printf("Failed external malloc\n");

		return nullptr;

	}


};


// template <size_t min_blocks_to_search>
// struct bitarr {

// 	uint64_t bits[64];

// 	__device__ void init(){
// 		memset(this, 255, sizeof(bitarr));
// 	}


// 	__device__ bool malloc(){


// 		//first, need to determine which threads in my warp are participating
// 		cg::coalesced_group active_threads = coalesced_threads();

// 		hasher = poggers::hash_schemes::murmurHasher<int, 1>;
// 		//give a randomish seed


// 		//hasher.init(clock64());
// 		hasher.init(14);

// 		int size = active_threads.size();


// 		for (int i = 0; i < min_blocks_to_search; i+=size){

// 			uint64_t my_address = hasher.hash(active_threads.thread_rank()) % 64;

// 			uit64_t my_bits = poggers::utils::ldca(bits+my_address);

// 			//ballot if we think my bits could fulfill all allocations
// 			int largest = cg::reduce(active_threads, __popcll(my_bits), cg::greater<int>());



// 			bool ballot = (__popcll(my_bits) == largest || __popcll(my_bits) >= size);

// 			auto ballot_result = active_threads.ballot(ballot);

// 			while (ballot_result){

// 	       			ballot = false;

// 	       			const auto leader = __ffs(ballot_result)-1;

// 	       			uint64_t working_bits = active_threads.shfl(my_bits, leader);

// 	       			//we need exactly this many
// 	       			int desired = __popcll(working_bits) - size;

// 	       			//everyone try a mask for the working bits
// 	       			int shift = size + __ffsll(working_bits) + active_threads.thread_rank();

// 	       			while (true){



// 	       				uint64_t bitmask = FULL_BITMASK >> (shift);

// 	       				bool success_ballot;
// 	       				if (bitmask & working_bits == desired){

// 	       				}

// 	       			}

	  

//        				//if leader succeeds return
//        				if (insert_tile.ballot(ballot)){
//        					return true;
//        				}
	       			

// 	       			//if we made it here no successes, decrement leader
// 	       			ballot_result  ^= 1UL << leader;

// 	       			//printf("Stalling in insert_into_bucket\n");

// 	       	}

	       		




// 		}



// 	}


// };




}

}


#endif //GPU_BLOCK_