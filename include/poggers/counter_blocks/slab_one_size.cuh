#ifndef SLAB_ONE_SIZE
#define SLAB_ONE_SIZE


#include <cuda.h>
#include <cuda_runtime_api.h>


#include <poggers/counter_blocks/block.cuh>
#include <poggers/counter_blocks/block_storage.cuh>
#include <poggers/counter_blocks/one_size_allocator.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define SLAB_ONE_SIZE_MAX_ATTEMPTS 50


#define SLAB_DEBUG_ARRAY 0

#define SLAB_ONE_SIZE_DEBUG 0

namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace beta {


namespace allocators { 



#if SLAB_DEBUG_ARRAY

template <typename allocator>
__global__ void check_block_kernel(allocator * alloc, uint64_t num_blocks){


	uint64_t tid =threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >=num_blocks) return;


	alloc->check_block(tid);

}


#endif



template <int extra_blocks>
struct one_size_slab_allocator {


	using my_type = one_size_slab_allocator<extra_blocks>;

	//doesn't seem necessary tbh
	//uint64_t offset_size;
	uint64_t offset_size;
	uint64_t num_blocks;
	one_size_allocator * block_allocator;
	//one_size_allocator * mem_alloc;
	char * extra_memory;

	pinned_block_storages<extra_blocks> * malloc_containers;

	#if SLAB_DEBUG_ARRAY
	uint64_t * debug_array;
	#endif


	//add hash table type here.
	//map hashes to bytes?


	static __host__ my_type * generate_on_device(uint64_t num_allocs, uint64_t ext_size){


	my_type * host_version;

	cudaMallocHost((void **)&host_version, sizeof(my_type));

	host_version->offset_size = ext_size;

	uint64_t num_pinned_blocks = (num_allocs-1)/4096+1;

	host_version->num_blocks = num_pinned_blocks;

	host_version->block_allocator = one_size_allocator::generate_on_device(num_pinned_blocks, sizeof(Block), 17);

    //host_version->mem_allocator = one_size_allocator::generate_on_device(num_pinned_blocks, 4096*ext_size, 1324);

	char * host_ptr_ext_mem;
	cudaMalloc((void **)&host_ptr_ext_mem, num_pinned_blocks*ext_size*4096);

	if (host_ptr_ext_mem == nullptr){
		throw std::runtime_error("main malloc buffer failed to be acquired.\n");
	}

	host_version->extra_memory = host_ptr_ext_mem;

 	host_version->malloc_containers = pinned_block_storages<extra_blocks>::generate_on_device(host_version->block_allocator);

	#if SLAB_DEBUG_ARRAY
	
 	uint64_t * debug_array_ptr;

 	cudaMalloc((void **)&debug_array_ptr, sizeof(uint64_t)*num_pinned_blocks);

 	cudaMemset(debug_array_ptr, 0, sizeof(uint64_t)*num_pinned_blocks);

 	host_version->debug_array = debug_array_ptr;

	#endif



 	my_type * dev_version;

 	cudaMalloc((void **)&dev_version, sizeof(my_type));

 	cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

 	cudaFreeHost(host_version);

 	cudaDeviceSynchronize();

 	return dev_version;


	}


	static __host__ void free_on_device(my_type * dev_version){

		my_type * host_version;
		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		one_size_allocator::free_on_device(host_version->block_allocator);
		//one_size_allocator::free_on_device(host_version->mem_allocator);

		cudaFree(host_version->extra_memory);

		pinned_block_storages<extra_blocks>::free_on_device(host_version->malloc_containers);

		#if SLAB_DEBUG_ARRAY

			cudaFree(host_version->debug_array);

		#endif

		cudaFree(dev_version);

		cudaFreeHost(host_version);

		return;


	}

	//returns address universe.
	__device__ uint64_t get_largest_allocation_offset(){

		return block_allocator->get_largest_allocation()*4096;

	}



	//request an allocation
	//returns a void * to the allocation
	//or a nullptr if no allocations available.

	__device__ void * malloc(){

		//__shared__ warp_lock team_lock;


		
		block_storage<extra_blocks> * my_storage = malloc_containers->get_pinned_blocks();

		//declare block ahead of time, all threads copy from leader.
		Block * my_block;

   		int num_attempts = 0;

   		while (num_attempts < SLAB_ONE_SIZE_MAX_ATTEMPTS){

   			cg::coalesced_group coalesced_team = cg::coalesced_threads();



   			if (coalesced_team.thread_rank() == 0){
   				my_block = my_storage->get_primary();
   			}

   			//blocking
   			my_block = coalesced_team.shfl(my_block, 0);

   			if (my_block == nullptr){
   				//team.sync();

   				//printf("Bitarr empty\n");
   				num_attempts+=1;
   				continue;
   			}

   			uint64_t block_id = block_allocator->get_offset_from_address(my_block);

   			if (block_id >= num_blocks){
   				printf("Wrong block\n");
   			}

   			//get offset from block.
   			uint64_t allocation = my_block->block_malloc(coalesced_team);


   			if (allocation == ~0ULL){


   				int result = my_storage->pivot_primary(my_block);


   				if (result != -1){

   					//malloc and replace pivot slab

   					{
   						uint64_t slab_offset = block_allocator->get_offset();

   						if (slab_offset == one_size_allocator::fail()){

   							return nullptr;

   						}

   						Block * slab = (Block *) block_allocator->get_mem_from_offset(slab_offset);

   						slab->init();

   						//this seems to be the bug?
   						//uint64_t slab_buffer_offset = slab_offset*4096;



   						//slab->attach_allocation(slab_buffer_offset);

   						__threadfence();

   						//printf("Attaching buffa\n");
   						
   						if (!my_storage->attach_new_buffer(result, slab)){
   							#if SLAB_ONE_SIZE_DEBUG
   							printf("Bug attaching buffer\n");
   							#endif
   						}
   						


   					}

   				}


   			} else {

   				return (void *) (extra_memory + (block_id*4096 +allocation)*offset_size);


   			}

   			num_attempts+=1;


   		}


   		return nullptr;


	}




	__device__ uint64_t get_offset_from_ptr(void * ext_ptr){

		//first off cast to uint64_t

		uint64_t ext_as_bits = (uint64_t) ext_ptr;

		//now downshift and subtract

		ext_as_bits = ext_as_bits - (uint64_t) extra_memory;

		ext_as_bits = ext_as_bits/offset_size;

		return ext_as_bits;

	}

	__device__ void * get_ptr_from_offset(uint64_t offset){

		uint64_t ext_as_bits = offset*offset_size;

		uint64_t mem_offset = (uint64_t) extra_memory + ext_as_bits;

		return (void *) mem_offset;

	}

	__device__ uint64_t malloc_offset(){

		void * alloc = malloc();

		if (alloc == nullptr) return ~0ULL;

		return get_offset_from_ptr(alloc);


	}


	//in the one allocator scheme free is simplified - get the block and free
	//if the block we free to is unpinned, we can safely return the memory to the veb tree
	__device__ void free(void * ext_allocation){

		uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

		uint64_t slab_offset = allocation_offset/4096;


		//this is nonatomic- disable
		// if (block_allocator->query(slab_offset)){
		// 	printf("Slab %llu is attached before its time\n", slab_offset);
		// }

		Block * slab = (Block * )  block_allocator->get_mem_from_offset(slab_offset);

		if (slab->block_free()){

			//slabs that are marked unpinned cannot be reattached - therefore, this read succeeding guarantees correctness.

			//printf("Returning block\n");

			block_allocator->free_offset(slab_offset);

			#if SLAB_ONE_SIZE_DEBUG
			if (!block_allocator->query(slab_offset)){
				printf("Slab %llu failed to attach to the tree\n", slab_offset);
			}	
			#endif

			

		}

	}


	__device__ void free_count_misses(void * ext_allocation, uint64_t * misses){

		uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

		uint64_t slab_offset = allocation_offset/4096;


		// if (block_allocator->query(slab_offset)){
		// 	printf("Slab %llu is attached before its time\n", slab_offset);
		// }

		Block * slab = (Block * )  block_allocator->get_mem_from_offset(slab_offset);

		if (slab->block_free()){

			//slab may be available for free - no pins in this version, can always free
			//multiple people may unpin?
			
			block_allocator->free_offset(slab_offset);

			atomicAdd((unsigned long long int *)&misses[0], 1ULL);

			if (!block_allocator->query(slab_offset)){
				printf("Slab %llu failed to attach to the tree\n", slab_offset);
			}	

			

		}

	}


	#if SLAB_DEBUG_ARRAY

	__device__ void log_free(void * ext_allocation){


		uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

		uint64_t slab_offset = allocation_offset/4096;


		Block * slab = (Block * )  block_allocator->get_mem_from_offset(slab_offset);


		if (!slab->belongs_to_block(allocation_offset)){
			printf("Bug in log_free.\n");
		}


		atomicAdd((unsigned long long int *)&debug_array[slab_offset], 1ULL);


		

	}


	#endif



	__host__ one_size_allocator * get_block_allocator_host(){

		my_type * host_full_alloc;

		cudaMallocHost((void **)&host_full_alloc, sizeof(my_type));

		cudaMemcpy(host_full_alloc, this, sizeof(my_type), cudaMemcpyDeviceToHost);

		one_size_allocator * host_block_allocator;

		cudaMallocHost((void **)&host_block_allocator, sizeof(one_size_allocator));

		cudaMemcpy(host_block_allocator, host_full_alloc->block_allocator, sizeof(one_size_allocator), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFreeHost(host_full_alloc);

		return host_block_allocator;

	}

	//report number of single size allocations available. 
	__host__ uint64_t report_fill(){


		one_size_allocator * host_block_allocator = get_block_allocator_host();
		

		uint64_t fill = host_block_allocator->report_fill();

		cudaFreeHost(host_block_allocator);

		return fill;

	}


	__host__ uint64_t report_max(){


		one_size_allocator * host_block_allocator = get_block_allocator_host();

		uint64_t max_fill = host_block_allocator->report_max();

		cudaFreeHost(host_block_allocator);

		return max_fill;

	}




};


}

}


#endif //GPU_BLOCK_