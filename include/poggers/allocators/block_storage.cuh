#ifndef BETA_BLOCK_STORAGE
#define BETA_BLOCK_STORAGE


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

#include <poggers/allocators/offset_slab.cuh>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>


#define SLAB_PRINT_DEBUG 0


namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 


	__global__ void beta_set_block_bitarrs(offset_alloc_bitarr ** blocks, uint64_t num_blocks){
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

		offset_alloc_bitarr ** blocks;

		static __host__ per_size_pinned_blocks * generate_on_device(uint64_t num_blocks){

			if (num_blocks == 0) num_blocks = 1;

			per_size_pinned_blocks * host_version = poggers::utils::get_host_version<per_size_pinned_blocks>();

			host_version->block_bitmap = malloc_bitarr::generate_on_device(num_blocks, false);

			host_version->blocks = poggers::utils::get_device_version<offset_alloc_bitarr *>(num_blocks);

			beta_set_block_bitarrs<<<(num_blocks-1)/512+1,512>>>(host_version->blocks, num_blocks);

			return poggers::utils::move_to_device<per_size_pinned_blocks>(host_version);


		}


		static __host__ void free_on_device(per_size_pinned_blocks * dev_version){

			per_size_pinned_blocks * host_version = poggers::utils::move_to_host<per_size_pinned_blocks>(dev_version);

			malloc_bitarr::free_on_device(host_version->block_bitmap);

			cudaFree(host_version->blocks);

			cudaFreeHost(host_version);

		}

		__device__ offset_alloc_bitarr * get_my_block(){



			int my_smid = poggers::utils::get_smid() % num_blocks;


			return blocks[my_smid];

						
			
		}


		__device__ bool swap_out_block(offset_alloc_bitarr * block){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return (atomicCAS((unsigned long long int *)&blocks[my_smid], (unsigned long long  int )block, 0ULL) == (unsigned long long int) block);


		}

		__device__ bool swap_out_nullptr(offset_alloc_bitarr * block){

			int my_smid = poggers::utils::get_smid() % num_blocks;

			return (atomicCAS((unsigned long long int *)&blocks[my_smid], 0ULL, (unsigned long long  int )block) == 0ULL);

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
	struct pinned_block_container {

		using my_type = pinned_block_container<smallest, biggest>;

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