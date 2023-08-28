#ifndef BETA_CONTEXT
#define BETA_CONTEXT
// Betta, the block-based extending-tree thread allocaotor, made by Hunter McCoy
// (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
//  portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/beta/thread_storage.cuh>
#include <poggers/beta/warp_lock.cuh>

namespace beta {

namespace allocators {

// Allocator context components that live in shared memory.
// Context is constructed simultaneously by all threads when entering a kernel.

// open context loads the context from the thread storage and boots the locks
// close context synchronizes and closes the system when the block is done.

// context contains all of the things needed inside of the main alloc loop
//  these are
//  1) block local locking to prevent over-subscription of storage containers
//  2) reference to thread storage

// context doesn't have to be closed as everything here is either atomically
// modified or local scope.
struct context {
  warp_lock local_lock;
  thread_storage* external_storage;

  __device__ context() {}
  __device__ ~context() {}

  __device__ void open_context(pinned_thread_storage* storages) {
    local_lock.init();

    external_storage = storages->get_thread_storage();

    __syncthreads();
  }

  __device__ void init_context_lock_only() {
    local_lock.init();

    __syncthreads();
  }

  __device__ warp_lock* get_local_lock() { return &local_lock; }

  __device__ thread_storage* get_local_storage() { return external_storage; }
};

// this version of context has an issue when loading new blocks
// context is not this context is no faster in testing.
//  struct context {

// 	warp_lock local_lock;

// 	thread_storage local_storage;
// 	thread_storage * external_storage;

// 	uint counter;

// 	uint64_t_bitarr lock_bits;

// 	//doesn't get triggered
// 	__device__ ~context(){
// 		printf("Thread %llu observes context close\n",
// poggers::utils::get_tid());
// 	}

// 	__device__ void open_context(thread_storage * ext_storage){

// 		local_lock.init();

// 		counter = 0;

// 		poggers::utils::cooperative_copy(&local_storage, ext_storage);

// 		external_storage = ext_storage;

// 		__syncthreads();

// 	}

// 	__device__ warp_lock * get_local_lock(){
// 		return &local_lock;
// 	}

// 	__device__ thread_storage * get_local_storage(){
// 		return &local_storage;
// 	}

// 	__device__ void close_context(){

// 		uint old = atomicAdd((unsigned int *)&counter, 1U);

// 		if (old == (blockDim.x-1)){
// 			//printf("Closing context for block %u\n", blockIdx.x);

// 			//*external_storage = local_storage;
// 		}

// 	}

// };

// struct scoped_context {

// 	context * local_context;

// 	__device__ scoped_context(context * ext_context){

// 		local_context = ext_context;

// 		// #ifdef __CUDA_ARCH__
// 		// printf("Thread %llu entered scoped context\n",
// poggers::utils::get_tid());
// 		// #endif
// 	}

// 	__device__ ~scoped_context(){

// 		local_context->close_context();
// 		//printf("Thread %llu exited scoped context\n",
// poggers::utils::get_tid());
// 	}

// };

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard