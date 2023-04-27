#ifndef ALLOC_TABLE
#define ALLOC_TABLE
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
#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/sub_veb.cuh>
#include <poggers/allocators/veb.cuh>
#include <poggers/allocators/offset_slab.cuh>

#include <poggers/allocators/malloc_bitarr.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from chunk_tree.

__global__ void betta_init_counters_kernel(unsigned int * counters, uint64_t num_segments, uint64_t blocks_per_segment){


	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_segments) return;

	counters[tid] = blocks_per_segment;

}


template<uint64_t bytes_per_segment, uint64_t min_size>
struct alloc_table {


	using my_type = alloc_table<bytes_per_segment, min_size>;

	//triplicate arrays

	//1) who owns it
	uint16_t * chunk_ids;

	//2) how much is paritioned
	malloc_bitarr ** segment_metadata;

	//3) what are the individual allocations.
	offset_alloc_bitarr * blocks;

	unsigned int * counters;

	char * memory;
	

	uint64_t num_segments;

	uint64_t blocks_per_segment;

	static __host__ my_type * generate_on_device(uint64_t max_bytes){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));


		uint64_t num_segments = poggers::utils::get_max_chunks<bytes_per_segment>(max_bytes);

		printf("Booting memory table with %llu chunks\n", num_segments);

		uint16_t * ext_chunks;

		cudaMalloc((void **)&ext_chunks, sizeof(uint16_t)*num_segments);


		cudaMemset(ext_chunks, ~0U, sizeof(uint16_t)*num_segments);

		host_version->chunk_ids = ext_chunks;

		host_version->num_segments = num_segments;


		//init blocks

		uint64_t blocks_per_segment = bytes_per_segment/(min_size*4096);

		offset_alloc_bitarr * ext_blocks;

		cudaMalloc((void **)&ext_blocks, sizeof(offset_alloc_bitarr)*blocks_per_segment*num_segments);


		cudaMemset(ext_blocks, 0U, sizeof(offset_alloc_bitarr)*(num_segments*blocks_per_segment));

		host_version->blocks = ext_blocks;

		host_version->blocks_per_segment = blocks_per_segment;

		host_version->memory = poggers::utils::get_device_version<char>(bytes_per_segment*num_segments);


		//end of blocks

		//start of metadata


		malloc_bitarr ** host_tree_array = poggers::utils::get_host_version<malloc_bitarr *>(num_segments);

		for (uint64_t i = 0; i < num_segments; i++){

			//this has high space usage, mark as todo to fix.
			host_tree_array[i] = malloc_bitarr::generate_on_device(blocks_per_segment, true);

		}

		host_version->segment_metadata = poggers::utils::move_to_device<malloc_bitarr *>(host_tree_array, num_segments);

		//end of metadata trees

		//start of counters

		host_version->counters = poggers::utils::get_device_version<unsigned int>(num_segments);

		betta_init_counters_kernel<<<(num_segments-1)/512+1,512>>>(host_version->counters, num_segments, blocks_per_segment);

		//cudaMemset(host_version->counters, 0, sizeof(unsigned int)*num_segments);

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

		malloc_bitarr ** host_trees = poggers::utils::move_to_host<malloc_bitarr *>(host_version->segment_metadata, host_version->num_segments);

		for (uint64_t i=0; i< host_version->num_segments; i++){
			malloc_bitarr::free_on_device(host_trees[i]);
		}

		cudaFree(host_version->blocks);

		cudaFree(host_version->chunk_ids);

		cudaFree(host_version->memory);

		cudaFree(host_version->counters);

		cudaFree(dev_version);

		cudaFreeHost(host_version);


	}

	//register a tree component
	__device__ void register_tree(uint64_t segment, uint16_t id){

		if (segment >= num_segments){
			printf("Chunk issue: %llu > %llu\n", segment, num_segments);
		}

		chunk_ids[segment] = id;

	}

	//register a segment from the table.
	__device__ void register_size(uint64_t segment, uint16_t size){

		if (segment >= num_segments){
			printf("Chunk issue\n");
		}

		size+=16;

		chunk_ids[segment] = size;

	}


	__device__ char * get_segment_memory_start(uint64_t segment){

		return memory + bytes_per_segment*segment;

	}


	__device__ bool setup_segment(uint64_t segment, uint16_t tree_id){

		//this serves to lock the segment
		//if people are fucking with it you can't claim.
		if (atomicCAS(&counters[segment], blocks_per_segment, 2*blocks_per_segment) != blocks_per_segment){
			//need to abort!
			printf("Failed to claim segment %lu for tree %u\n", segment, tree_id);
			return false;
		}

		uint64_t tree_alloc_size = get_tree_alloc_size(tree_id);

		chunk_ids[segment] = tree_id;

		//gate to init is init_new_universe
		

		uint64_t num_blocks = bytes_per_segment/(tree_alloc_size*4096);

		segment_metadata[segment]->init_new_universe(num_blocks);

		__threadfence();

		return setup_blocks(segment, num_blocks);

	}

	__device__ bool setup_blocks(uint64_t segment, uint64_t num_blocks){


		__threadfence();

		uint64_t base_offset = blocks_per_segment*segment;

		for (uint64_t i = 0; i < num_blocks; i++){


			offset_alloc_bitarr * block = &blocks[base_offset+i];

			block->init();

			uint64_t global_offset = get_global_block_offset(block);

			// if (global_offset != base_offset+i){
			// 	printf("Issue in setting blocks\n");
			// }

			// if (global_offset*4096 != 4096*(base_offset+i)){
			// 	printf("Rounding issue\n");
			// }

			block->attach_allocation(global_offset*4096);

			__threadfence();





		}

		return (atomicCAS(&counters[segment], 2*blocks_per_segment, 0) == 2*blocks_per_segment);



	}


	__device__ bool set_tree_id(uint64_t segment, uint16_t tree_id){


		return (atomicCAS((unsigned short int *)&chunk_ids[segment],  (unsigned short int) ~0U, (unsigned short int)tree_id) == (unsigned short int) ~0U);
	}


	__device__ uint16_t read_tree_id(uint64_t segment){

		return atomicCAS((unsigned short int *)&chunk_ids[segment],  (unsigned short int) ~0U, (unsigned short int) ~0U);

	}

	__device__ bool reset_tree_id(uint64_t segment, uint16_t tree_id){

		return (atomicCAS((unsigned short int *)&chunk_ids[segment], (unsigned short int)tree_id, (unsigned short int) ~0U) == (unsigned short int) tree_id);
	}

	//in the near future this will be allowed to fail
	__device__ offset_alloc_bitarr * get_block(uint64_t segment_id, uint16_t tree_id, bool &empty){

		//else valid
		//get bit

		//I know a bit is available.

		// uint64_t block_id = malloc_bitarr::fail();

		// while (block_id == malloc_bitarr::fail()){
		// 	block_id = segment_metadata[segment_id]->malloc();
		// }

		// return &blocks[segment_id*blocks_per_segment+block_id];

		//last thread to pull from a block is obligated to unset the block in the sub tree.
		//empty = (my_count == 0);

		uint16_t global_tree_id = read_tree_id(segment_id);



		if (tree_id != global_tree_id){
			//printf("Tree load err in %llu, %u != %u\n", segment_id, tree_id, global_tree_id);

			empty = false;
			return nullptr;
		}

		uint64_t num_blocks = get_blocks_per_segment(global_tree_id);

		int my_count = atomicAdd(&counters[segment_id], 1);

		if (my_count >= num_blocks){
			//this is fine
			//printf("Count failure in segment %llu\n", segment_id);
			atomicSub(&counters[segment_id], 1);

			return nullptr;
		}


		auto block_id = segment_metadata[segment_id]->malloc();

		if (block_id == malloc_bitarr::fail()){

			//this shouldn't ever happen *but* it can
			//This means that we had a full reset,
			printf("Shouldn't ever happen\n");

			atomicSub(&counters[segment_id], 1);
			empty = false;
			return nullptr;

		}



		if (my_count == (num_blocks-1)){
			empty = true;
		}

		

		return &blocks[segment_id*blocks_per_segment+block_id];

		// } else {

		// 	printf("Failed to get ID\n");

		// 	//block_id = segment_metadata[segment_id]->malloc();

		// }

		//return nullptr;
	}


	__device__ uint64_t get_segment_from_block_ptr(offset_alloc_bitarr * block){


		//this returns the stride in blocks
		uint64_t offset = (block-blocks);

		return offset/blocks_per_segment;

	}

	__device__ int get_relative_block_offset(offset_alloc_bitarr * block){


		uint64_t offset = (block-blocks);

		return offset % blocks_per_segment;

	}

	__device__ offset_alloc_bitarr * get_block_from_ptr(void * ptr){


	}

	__device__ uint64_t get_segment_from_ptr(void * ptr){

		uint64_t offset = ((char *) ptr) - memory;

		return offset/bytes_per_segment;

	}

	__device__ int get_tree_from_segment(uint64_t segment){

		return chunk_ids[segment];

	}

	static __host__ __device__ uint64_t get_p2_from_index(int index){


		return (1ULL) << index;


	}

	static __host__ __device__ uint64_t get_tree_alloc_size(uint16_t tree){

		//scales up by smallest.
		return min_size * get_p2_from_index(tree);

	}


	__device__ uint64_t get_global_block_offset(offset_alloc_bitarr * block){
		return block - blocks;
	}

	static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree){

		uint64_t tree_alloc_size = get_tree_alloc_size(tree);

		return bytes_per_segment/(tree_alloc_size*4096);
	}

	//free block, returns true if this block was the last section needed.
	__device__ bool free_block(offset_alloc_bitarr * block_ptr, bool &need_to_reregister){

		need_to_reregister = false;

		uint64_t segment = get_segment_from_block_ptr(block_ptr);

		int relative_offset = get_relative_block_offset(block_ptr);

		uint16_t tree_id = chunk_ids[segment];

		uint64_t tree_alloc_size = get_tree_alloc_size(tree_id);

		uint64_t num_blocks = bytes_per_segment/(tree_alloc_size*4096);

		//first, add back to tree


		int old_count = atomicSub(&counters[segment], 1);

		//printf("Old count for offset %d is %d\n", relative_offset, old_count);


		if (!segment_metadata[segment]->insert(relative_offset)){
			printf("Double free in tree %llu at address %d\n", segment, relative_offset);
		}


		if (old_count == 1){




			//can maybe free section.
			//attempt CAS
			//on success, you are the exclusive owner of the segment.

			return (atomicCAS(&counters[segment], 0, blocks_per_segment) == 0);

		}


		return false;


	}


};



}

}


#endif //End of VEB guard