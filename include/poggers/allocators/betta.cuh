#ifndef BETTA_ALLOCATOR
#define BETTA_ALLOCATOR
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
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/one_size_allocator.cuh>

#include <poggers/allocators/offset_slab.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from segment_tree.

using namespace poggers::utils;


template<uint64_t bytes_per_segment, uint64_t smallest, uint64_t biggest>
struct betta_allocator {


	using my_type = betta_allocator<bytes_per_segment, smallest, biggest>;
	using sub_tree_type = extending_veb_allocator_nosize<bytes_per_segment, 5>;

	one_size_allocator * bit_tree;
	one_size_allocator * segment_tree;

	alloc_table<bytes_per_segment> * table;

	sub_tree_type ** sub_trees;

	int num_trees;

	int smallest_bits;

	static __host__ my_type * generate_on_device(uint64_t max_bytes, uint64_t seed){


		my_type * host_version = get_host_version<my_type>();


		//plug in to get max chunks
		uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

		host_version->segment_tree = one_size_allocator::generate_on_device(max_chunks, bytes_per_segment, seed);


		//estimate the max_bits
		uint64_t num_bits = bytes_per_segment/(4096*smallest);

		uint64_t num_bytes = 0;

		//this underestimates the total #bytes needed.

		do {

			printf("Bits is %llu, bytes is %llu\n", num_bits, num_bytes);

			num_bytes += ((num_bits -1)/64+1)*8;

			num_bits = num_bits/64;
		} while (num_bits > 64);

		num_bytes += 8+num_bits*sizeof(offset_alloc_bitarr);


		printf("Final bits is %llu, bytes is %llu\n", num_bits, num_bytes);

		//need to verify, but this should be sufficient for all sizes.
		host_version->bit_tree = one_size_allocator::generate_on_device(max_chunks, num_bytes, seed);

		printf("Each bit tree array gets %llu\n", num_bytes);

		uint64_t num_trees = get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest)+1;

		host_version->smallest_bits = get_first_bit_bigger(smallest);

		host_version->num_trees = num_trees;

		printf("Booting %llu trees\n", num_trees);

		sub_tree_type ** ext_sub_trees = get_host_version<sub_tree_type *>(num_trees);

		for (int i = 0; i < num_trees; i++){

			//debugging
			sub_tree_type * temp_tree = sub_tree_type::generate_on_device(get_p2_from_index(get_first_bit_bigger(smallest)+i), seed+i, max_bytes);

			ext_sub_trees[i] = temp_tree;

		}

		host_version->sub_trees = move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

		printf("Host sub tree %lx, dev %lx\n", (uint64_t) ext_sub_trees, (uint64_t) host_version->sub_trees);

		host_version->table = alloc_table<bytes_per_segment>::generate_on_device(max_bytes, host_version->segment_tree->get_allocator_memory_start());

		return move_to_device(host_version);

	}


	//return the index of the largest bit set
	static __host__ __device__ int get_first_bit_bigger(uint64_t counter){

	//	if (__builtin_popcountll(counter) == 1){

			//0th bit would give 63

			//63rd bit would give 0

		#ifndef __CUDA_ARCH__

			return 63 - __builtin_clzll(counter) + (__builtin_popcountll(counter) != 1);

		#else 

			return 63 - __clzll(counter) + (__popcll(counter) != 1);

		#endif

	}


	static __host__ __device__ uint64_t get_p2_from_index(int index){


		return (1ULL) << index;


	}

	static __host__ void free_on_device(my_type * dev_version){

		//this frees dev version.
		my_type * host_version = move_to_host<my_type>(dev_version);

		uint64_t num_trees = get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest)+1;


		sub_tree_type ** host_subtrees = move_to_host<sub_tree_type *>(host_version->sub_trees, num_trees);


		for (int i=0; i < num_trees; i++){

			sub_tree_type::free_on_device(host_subtrees[i]);

		}

		alloc_table<bytes_per_segment>::free_on_device(host_version->table);

		one_size_allocator::free_on_device(host_version->bit_tree);

		one_size_allocator::free_on_device(host_version->segment_tree);

		cudaFreeHost(host_subtrees);

		cudaFreeHost(host_version);


	}

	__device__ inline uint64_t snap_pointer_to_block(void * ext_ptr){


		uint64_t snapped_offset =  ((uint64_t) ext_ptr) / bytes_per_segment;

		return snapped_offset;


	}


	//for small allocations, attempt to grab locks.
	__device__ void * malloc(uint64_t bytes_needed){
 
		int allocator_needed = get_first_bit_bigger(smallest) - smallest_bits;

		if (allocator_needed >= num_trees){

			//get big allocation
			printf("Larger allocations not yet implemented\n");

			return nullptr;

		} else {


			

			return nullptr;

		}


	}

	//get a new segment for a tree!
	__device__ bool gather_new_segment(int tree){

		void * bits = bit_tree->malloc();

		if (bits == nullptr){
			return false;
		}

		void * memory_segment = segment_tree->malloc();

		if (memory_segment == nullptr){
			bit_tree->free(bits);

			return false;
		}

		//otherwise, both initialized
		//register segment
		table->register_tree_ptr(memory_segment, tree);

		//and add to tree
		sub_trees[tree]->register_new_segment(bits, memory_segment);

	}





};



}

}


#endif //End of VEB guard