#ifndef BETA_SHARED_BLOCK_STORAGE
#define BETA_SHARED_BLOCK_STORAGE


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include <poggers/allocators/malloc_bitarr.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <poggers/beta/block.cuh>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>


#define SLAB_PRINT_DEBUG 0


namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace beta {


namespace allocators { 


	//should these start initialized? I can try it.
	__global__ void beta_set_block_bitarrs(block ** blocks, uint64_t num_blocks){
		uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

		if (tid >= num_blocks) return;

		blocks[tid] = nullptr;
	}

	//per size pinned blocks have one block per size (wow).
	//in your lifetime.
	//read your block.
	//if 
	struct per_size_pinned_blocks {

		uint64_t num_blocks;

		malloc_bitarr * block_bitmap;

		block ** blocks;

		static __host__ per_size_pinned_blocks * generate_on_device(uint64_t num_blocks){

			if (num_blocks == 0) num_blocks = 1;

			per_size_pinned_blocks * host_version = poggers::utils::get_host_version<per_size_pinned_blocks>();

			host_version->block_bitmap = malloc_bitarr::generate_on_device(num_blocks, false);

			host_version->blocks = poggers::utils::get_device_version<block *>(num_blocks);

			host_version->num_blocks = num_blocks;

			beta_set_block_bitarrs<<<(num_blocks-1)/512+1,512>>>(host_version->blocks, num_blocks);

			return poggers::utils::move_to_device<per_size_pinned_blocks>(host_version);


		}


		static __host__ void free_on_device(per_size_pinned_blocks * dev_version){

			per_size_pinned_blocks * host_version = poggers::utils::move_to_host<per_size_pinned_blocks>(dev_version);

			malloc_bitarr::free_on_device(host_version->block_bitmap);

			cudaFree(host_version->blocks);

			cudaFreeHost(host_version);

		}

		__device__ block * get_my_block(){



			int my_smid = poggers::utils::get_smid() % num_blocks;


			return blocks[my_smid];

						
			
		}


		__device__ block * get_alt_block(){


			int my_smid = poggers::utils::get_smid();

			my_smid = my_smid*my_smid % num_blocks;

			return blocks[my_smid];

		}


		__device__ bool swap_out_block(block * block_to_swap){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return (atomicCAS((unsigned long long int *)&blocks[my_smid], (unsigned long long  int )block_to_swap, 0ULL) == (unsigned long long int) block_to_swap);


		}

		__device__ bool replace_block(block * old_block, block * new_block){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return (atomicCAS((unsigned long long int *)&blocks[my_smid], (unsigned long long  int )old_block,  (unsigned long long  int )new_block) == (unsigned long long int) old_block);


		}

		__device__ bool swap_out_nullptr(block * block_to_swap){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return (atomicCAS((unsigned long long int *)&blocks[my_smid], 0ULL, (unsigned long long  int )block_to_swap) == 0ULL);

		}


		__device__ bool lock_my_block(){
			int my_smid = poggers::utils::get_smid() % num_blocks;

			return block_bitmap->insert(my_smid);
		}


		__device__ bool unlock_my_block(){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return block_bitmap->remove(my_smid);

		}


	};


	//container has one of these per size. 
	template <uint64_t smallest, uint64_t biggest>
	struct pinned_shared_blocks {

		using my_type = pinned_shared_blocks<smallest, biggest>;

		per_size_pinned_blocks ** block_containers;

		static __host__ my_type * generate_on_device(uint64_t blocks_per_segment){


			my_type * host_version = poggers::utils::get_host_version<my_type>();

			uint64_t num_trees = poggers::utils::get_first_bit_bigger(biggest) - poggers::utils::get_first_bit_bigger(smallest)+1;


			per_size_pinned_blocks ** host_block_containers = poggers::utils::get_host_version<per_size_pinned_blocks *>(num_trees);

			for (uint64_t i = 0; i< num_trees; i++){

				host_block_containers[i] = per_size_pinned_blocks::generate_on_device(blocks_per_segment);

				blocks_per_segment = blocks_per_segment/2;

			}

			host_version->block_containers = poggers::utils::move_to_device<per_size_pinned_blocks *>(host_block_containers, num_trees);

			return poggers::utils::move_to_device<my_type>(host_version);


		}

		static __host__ void free_on_device(my_type * dev_version){

			my_type * host_version = poggers::utils::move_to_host<my_type>(dev_version);

			uint64_t num_trees = poggers::utils::get_first_bit_bigger(biggest) - poggers::utils::get_first_bit_bigger(smallest)+1;

			per_size_pinned_blocks ** host_block_containers = poggers::utils::move_to_host<per_size_pinned_blocks *>(host_version->block_containers, num_trees);

			for (uint64_t i = 0; i < num_trees; i++){

				per_size_pinned_blocks::free_on_device(host_block_containers[i]);

			}

			cudaFreeHost(host_version);

			cudaFreeHost(host_block_containers);

		}


		__device__ per_size_pinned_blocks * get_tree_local_blocks(int tree){

			return block_containers[tree];

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