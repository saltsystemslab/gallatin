#ifndef BLOCK_TABLE
#define BLOCK_TABLE
//A CUDA implementation of the alloc table, made by Hunter McCoy (hunter@cs.utah.edu)
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
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/allocators/offset_slab.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from chunk_tree.


//the block table is the controller for all blocks
//contains controls to lookup
//all these do is store and return offsets, swap to tree to think about it.
//block_metadata_table controls metadata.
template<uint64_t bytes_per_segment, uint64_t min_size>
struct block_table {


	using my_type = block_table<bytes_per_segment, min_size>;

	offset_alloc_bitarr * blocks;

	uint64_t max_blocks;

	uint64_t blocks_per_segment;

	static __host__ my_type * generate_on_device(void * ext_starting_address){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));


		uint64_t num_chunks = poggers::utils::get_max_chunks<bytes_per_segment>();

		uint64_t blocks_per_segment = bytes_per_segment/(min_size*4096);

		printf("Booting block table with %llu chunks, %llu blocks per segment\n", num_chunks, blocks_per_segment);

		offset_alloc_bitarr * ext_blocks;

		cudaMalloc((void **)&ext_blocks, sizeof(offset_alloc_bitarr)*(num_chunks*blocks_per_segment));


		cudaMemset(ext_blocks, 0U, sizeof(offset_alloc_bitarr)*(num_chunks*blocks_per_segment));

		host_version->blocks = ext_blocks;

		host_version->max_blocks = (num_chunks*blocks_per_segment);

		host_version->blocks_per_segment = blocks_per_segment;

		my_type * dev_version;


		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_version);


		return dev_version;


	}

	static __host__ my_type * generate_on_device(uint64_t max_bytes, void * ext_starting_address){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));


		uint64_t num_chunks = poggers::utils::get_max_chunks<bytes_per_segment>(max_bytes);

		uint64_t blocks_per_segment = bytes_per_segment/(min_size*4096);

		printf("Booting block table with %llu chunks, %llu blocks per segment\n", num_chunks, blocks_per_segment);

		offset_alloc_bitarr * ext_blocks;

		cudaMalloc((void **)&ext_blocks, sizeof(offset_alloc_bitarr)*(num_chunks*blocks_per_segment));


		cudaMemset(ext_blocks, 0U, sizeof(offset_alloc_bitarr)*(num_chunks*blocks_per_segment));

		host_version->blocks = ext_blocks;

		host_version->max_blocks = (num_chunks*blocks_per_segment);

		host_version->blocks_per_segment = blocks_per_segment;

		my_type * dev_version;


		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_version);


		return dev_version;



	}

	static __host__ void free_on_device(my_type * dev_version){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);


		cudaDeviceSynchronize();

		cudaFree(host_version->chunk_ids);

		cudaFree(dev_version);

		cudaFreeHost(host_version);


	}


	//functions needed...
	//get block given segment and local offset...
	//free to block given offset (tree segmenter handles casting void pointers.)

	//given a valid pointer
	__device__ offset_alloc_bitarr * get_block(uint64_t offset){

		if (offset >= max_blocks){
			printf("Asking for block %llu/%llu that does not exist\n", offset, max_blocks);
		}

		return blocks[offset];

	}


	__device__ offset_alloc_bitarr * get_block(uint64_t segment_id, uint64_t local_offset){

		uint64_t global_offset = segment_id*blocks_per_segment+local_offset;

		return get_block(global_offset);

	}

	//needs free to block




};



}

}


#endif //End of VEB guard