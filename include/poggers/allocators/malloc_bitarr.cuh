#ifndef MALLOC_BITARR_H
#define MALLOC_BITARR_H
//A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy (hunter@cs.utah.edu)
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


//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>


#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))


//cudaMemset is being weird
__global__ void malloc_bitarr_init_bits(uint64_t * bits, uint64_t items_in_universe){

	uint64_t tid = threadIdx.x +blockIdx.x*blockDim.x;

	if (tid >= items_in_universe) return;

	uint64_t high = tid/64;

	uint64_t low = tid % 64;

	atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low));

	//bits[tid] = ~(0ULL);

}

template <typename malloc_bitarr_type>
__global__ void bitarr_report_fill_kernel(malloc_bitarr_type * tree, uint64_t num_threads, uint64_t * fill_count){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_threads) return;


	uint64_t my_fill = __popcll(tree->bits[tid]);

	atomicAdd((unsigned long long int *)fill_count, (unsigned long long int) my_fill);


}

template <typename malloc_bitarr_type>
__global__ void report_missing_kernel(malloc_bitarr_type * tree){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid != 0) return;

	tree->dev_print_missing();
}



struct malloc_bitarr {

	uint64_t total_universe;
	uint64_t num_blocks;
	uint64_t * bits;

	//don't think this calculation is correct
	__host__ static malloc_bitarr * generate_on_device(uint64_t universe, bool set_bits){



		malloc_bitarr * host_tree;

		cudaMallocHost((void **)&host_tree, sizeof(malloc_bitarr));


		uint64_t num_blocks = (universe-1)/64+1;

		uint64_t * bits;

		cudaMalloc((void **)&bits, num_blocks*sizeof(uint64_t));

		cudaMemset(bits, 0, num_blocks*sizeof(uint64_t));

		if (set_bits){

			malloc_bitarr_init_bits<<<(universe-1)/512+1,512>>>(bits, universe);

		}


		//setup host structure
		host_tree->total_universe = universe;
		host_tree->bits = bits;
		host_tree->num_blocks = num_blocks;

		malloc_bitarr * dev_tree;
		cudaMalloc((void **)&dev_tree, sizeof(malloc_bitarr));


		cudaMemcpy(dev_tree, host_tree, sizeof(malloc_bitarr), cudaMemcpyHostToDevice);

		cudaFreeHost(host_tree);

		return dev_tree;


	}


	__host__ static void free_on_device(malloc_bitarr * dev_tree){


		malloc_bitarr * host_tree;

		cudaMallocHost((void **)&host_tree, sizeof(malloc_bitarr));

		cudaMemcpy(host_tree, dev_tree, sizeof(malloc_bitarr), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();


		cudaFree(host_tree->bits);

		cudaDeviceSynchronize();

		cudaFreeHost(host_tree);

		cudaFree(dev_tree);


	}


	__device__ uint64_t malloc(){

		__threadfence();

		for (int i = 0; i < num_blocks; i++){

			//try load wrong
			//uint64_t current_block = ~0ULL;
			//doesn't really change anything.
			poggers::utils::ldca(&bits[i]);

			while (current_block != 0ULL){

				int bit = __ffsll(current_block)-1;

				if (bit == -1) continue;

				current_block = atomicAnd((unsigned long long int *)&bits[i], ~SET_BIT_MASK(bit));

				if (current_block & SET_BIT_MASK(bit)){

					if (i*64+bit >= load_universe_atomic()){
						printf("Bug in bitarr: %llu > %llu\n", i*64+bit, load_universe_atomic());
						return fail();
					}
					return i*64 + bit;  
				}

			}


		}

		return fail();

	}



	__device__ bool insert(uint64_t set_bit){

		if (set_bit >= load_universe_atomic()){
			printf("inserting from illegal section, bit %llu in universe %llu\n", set_bit, load_universe_atomic());
		}


		uint64_t high = set_bit/64;

		uint64_t low = set_bit % 64;

		uint64_t old_state = atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low));

		//if old state is 0, now 1 - we are the first add.
		return !(old_state & SET_BIT_MASK(low));

	}

	__device__ bool remove(uint64_t set_bit){

		if (set_bit >= load_universe_atomic()){
			printf("removing from illegal section\n");
		}

		uint64_t high = set_bit/64;

		uint64_t low = set_bit % 64;

		uint64_t old_state = atomicAnd((unsigned long long int *)&bits[high], ~SET_BIT_MASK(low));

		return (old_state & SET_BIT_MASK(low));

	}

	__device__ bool atomic_query(uint64_t bit){

		uint64_t high = bit/64;

		uint64_t low = bit % 64;

		uint64_t old_state = atomicOr((unsigned long long int *)&bits[high], 0ULL);

		return old_state & SET_BIT_MASK(low);

	}


	__device__ uint64_t atomic_get_first(){


		for (uint64_t i = 0; i < num_blocks; i++){

			uint64_t old_state = atomicOr((unsigned long long int *)&bits[i], 0ULL);

			if (old_state != 0ULL){
				return i*64+ __ffsll(old_state)-1;
			}

		}

		return fail();

	}


	// __device__ uint64_t atomic_get_random(int rand){



	// }

	__device__ uint64_t load_universe_atomic(){

		return atomicOr((unsigned long long int *)&total_universe, 0ULL);
	}

	//wipe old bits
	__device__ void init_new_universe(uint64_t items_in_universe){


		uint64_t new_num_blocks = (items_in_universe - 1)/64+1;



		
		for (int i = 0; i < num_blocks; i++){
			atomicExch((unsigned long long int *)&bits[i], 0ULL);
		}

		//then reset

		for (uint64_t i = 0; i < new_num_blocks; i++){

			uint64_t leftover = items_in_universe - i*64;

			if (leftover > 64) leftover = 64;

			atomicOr((unsigned long long int *)&bits[i], BITMASK(leftover));


		}

		total_universe = items_in_universe;

		//num_blocks = new_num_blocks;


		__threadfence();

	}

	__device__ void zero_out(uint64_t items_in_universe){

		for (int i =0; i < num_blocks; i++){
			atomicExch((unsigned long long int *)&bits[i], 0ULL);
		}

		total_universe = items_in_universe;

		num_blocks = (items_in_universe -1)/64+1;

		__threadfence();

	}

	__device__ __host__ static uint64_t fail(){
		return ~0ULL;
	}


	
	__device__ uint64_t get_largest_allocation(){

		return total_universe;

	}

	__host__ uint64_t host_get_universe(){

		malloc_bitarr * host_version;

		cudaMallocHost((void **)&host_version, sizeof(malloc_bitarr));

		cudaMemcpy(host_version, this, sizeof(malloc_bitarr), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		uint64_t ret_value = host_version->total_universe;

		cudaFreeHost(host_version);

		return ret_value;

	}


	__host__ uint64_t report_fill(){


		uint64_t * fill_count;

		cudaMallocManaged((void **)&fill_count, sizeof(uint64_t));

		cudaDeviceSynchronize();

		fill_count[0] = 0;

		cudaDeviceSynchronize();

		uint64_t max_value = report_max();

		uint64_t num_threads = (max_value-1)/64+1;

		bitarr_report_fill_kernel<malloc_bitarr><<<(num_threads-1)/512+1, 512>>>(this, num_threads, fill_count);

		cudaDeviceSynchronize();

		uint64_t return_val = fill_count[0];

		cudaFree(fill_count);

		return return_val;


	}

	__device__ void dev_print_missing(){

		for (int i = 0; i < num_blocks; i++){

			for (int j = 0; j < 64; j++){

				if (i*64+j >= total_universe) return;

				if (!(bits[i] & SET_BIT_MASK(j))){
					printf("Missing %d\n", i*64+j);
				}

			}
			
		}
	}


	__host__ uint64_t report_max(){


		//return 1;
		return host_get_universe();

	}


	__host__ void print_missing(){

		report_missing_kernel<malloc_bitarr><<<1,1>>>(this);

		cudaDeviceSynchronize();
	}



};



}

}


#endif //End of VEB guard