#ifndef BETA_ALLOC_WITH_LOCKS 
#define BETA_ALLOC_WITH_LOCKS 
//Betta, the block-based extending-tree thread allocaotor, made by Hunter McCoy (hunter@cs.utah.edu)
//Copyright (C) 2023 by Hunter McCoy

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//and associated documentation files (the "Software"), to deal in the Software without restriction, 
//including without l> imitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
//LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//The alloc table is an array of uint64_t, uint64_t pairs that store



//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include <poggers/beta/warp_lock.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>




#define ALLOC_LOCK_DEBUG 0


namespace beta {

namespace allocators {



//Allocate with locks
//this initializes a shared warp lock to reduce contention on the blocks and prevent thread
//read-write hazards
//takes in the block (+block ID), along with the pinned storage from this Streaming Multiprocessor

//code runs in 4 phases
// 1) lock is acquired

// 2) coalesced team enters the lock, attempts to pull from prealloced thread storage
// 3) if this doesn't satisfy, pull from the requested block
// 4) leftovers go to the main func.

__device__ uint64_t alloc_with_locks(warp_lock * team_lock, uint64_t block_id, block * alloc_block, thread_storage * block_storage){

	uint64_t remainder = 0ULL;

	//__shared__ warp_lock team_lock;


	//1) stall while acquiring coalesced lock.
	while (true){

		cg::coalesced_group grouping = cg::coalesced_threads();

		bool ballot = false;

		if (grouping.thread_rank() == 0){	

			//one thread groups;

			ballot = team_lock->lock();

			
		}
		//team ballots, then exits as the CG cannot exist outside of this scope
		//we then recoalesce immediately after
		//I have verified that this does work.
		if (grouping.ballot(ballot)) break;

	}

	cg::coalesced_group in_lock = cg::coalesced_threads();

	__threadfence();

	//printf("Progressing %llu out of team lock\n", threadIdx.x+blockIdx.x*blockDim.x);



	//in lock is coalesced team;
	//2) attempt to grab an existing allocation.
	uint64_t allocation = (block_storage->malloc(in_lock, remainder));

	//allocation already has its upper bits set.
	uint64_t alloc_offset = (allocation - (allocation % 64));

	//if 100% of requests are satisfied, we are all returning, so one thread needs to drop lock.
	//this makes a clever assumption that if any request was not satisfied then no remainder left

	#if ALLOC_LOCK_DEBUG

	if (allocation == ~0ULL && remainder != 0){
		printf("Allocation could not be acquired but leftover is present\n");
	}

	#endif


	//scope out ballot to limit register use
	{

	bool ballot = (allocation != ~0ULL);

	if ( __popc(in_lock.ballot(ballot)) == in_lock.size()){

		if (in_lock.thread_rank() == 0){

			if (__popcll(remainder) > 0){
				block_storage->attach_buffer(alloc_offset, remainder);
			}

			team_lock->unlock();
		}

	}


	in_lock.sync();

	//failures shouldn't return, as they progress
	//success on the ballot means that an allocation was acquired for you
	//leftovers have been handled, so return. 
	if (ballot){

		if (allocation == (~0ULL -1)){
			printf("Error at first ballot\n");
		}

		return allocation;
	}


	}






	//re-coalesce to figure out who still needs an allocation.
	cg::coalesced_group remaining = cg::coalesced_threads();


	//3) alloc from existing block.
	allocation = alloc_block->block_malloc(remaining, remainder);

	alloc_offset = block_id*4096 + (allocation - (allocation % 64));


	if (__popcll(remainder) && (allocation == ~0ULL)){
		printf("Failed to distribute keys\n");
	}


	if (remaining.thread_rank() == 0){
	      
		  //only attempt to attach if not empty.
		  //if these conditions pass, we can guarantee that all threads were satisfied.
	      if (__popcll(remainder) > 0  && (allocation != ~0ULL)){
		      bool result = block_storage->attach_buffer(alloc_offset, remainder);

		      #if ALLOC_LOCK_DEBUG
		      if (!result){
		      	printf("Failed to attach - this is a bug\n");
		      }
		      #endif



	  		}
	      
	      team_lock->unlock();

	}

	__threadfence();


	if (allocation != ~0ULL){
		allocation = allocation+block_id*4096;
	}

	if (allocation == (~0ULL -1)){
		printf("Error at end\n");
	}

	return allocation;


}


}

}


#endif //End of VEB guard