#ifndef POGGERS_BITBUDDY
#define POGGERS_BITBUDDY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/templated_bitbuddy.cuh>
#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/allocators/slab.cuh>

#include <stdio.h>
#include <iostream>
#include "assert.h"
#include <vector>
#include <chrono>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;



//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 

	template <typename allocator_type>
	__global__ void init_simple_allocator(allocator_type * allocator,  int num_sms){


		uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

		if (tid < num_sms){

			allocator->locks = 0ULL;

			//bitbuddy is init, skip

			allocator->storages[tid].init();



		}

		if (tid == 0ULL) allocator->local_slabs.slab_markers = 0ULL;

		__syncthreads();

		// if (tid < 32){

		// 	allocator->load_new_slab();

		// }

		if (tid == 0){

			for (int i = 0; i < 32; i++){
				allocator->load_new_slab(i);
			}
		}


		__syncthreads();

		return;




	}

	//the final outcome of our experiments: one unified allocator for GPU
	//bitbuddy supplies high level allocations, 

	template <uint64_t alloc_size>
	struct simple_allocator_one_size {


		using my_type = simple_allocator_one_size<alloc_size>;

		using bitbuddy_type = bitbuddy_allocator<(1ULL << 12), 256>;

		//For now, lets reserve 4GB memory - make this template arg in full version
		//bytes desired = 4*1024*1024*1024;

		//2^32 in chunks of 2^14

		//2^18

		//TODO: make bitbuddy not use all the space lmao



		bitbuddy_type * my_bitbuddy;

		storage_bitmap * storages;

		//slab_retreiver * malloced_slabs;

		slab_storage<alloc_size> local_slabs;

		//set of helpful locks to stop bullshit
		uint64_t_bitarr locks;

		// __device__ void load_new_slab(){


		// 	alloc_bitarr * slab;

		// 	while (true){

		// 		slab = malloced_slabs->give_bitmap();

		// 		if (slab == nullptr){

		// 			//attempt to take lock
		// 			if (!(locks.set_index(0) & 1ULL)){

		// 				slab_retreiver * next_block = my_bitbuddy->malloc(sizeof(slab_retreiver));

		// 				void * memory_for_blocks = my_bitbuddy->malloc(alloc_size*4096*31);

		// 				if (next_block == nullptr || memory_for_blocks == nullptr){

		// 					printf("Buddy allocator out of space for allocations!\n");
		// 					return;
		// 				}



		// 			}	


		// 		}

		// 	}


		__device__ void load_new_slab(int index){


			alloc_bitarr * slab = (alloc_bitarr *) my_bitbuddy->malloc(sizeof(alloc_bitarr));


			assert(((uint64_t) slab) % size(alloc_bitarr) == ((uint64_t) slab));

			void * memory_for_blocks = my_bitbuddy->malloc(alloc_size*4096);

			if (slab == nullptr || memory_for_blocks == nullptr){

				printf("Buddy allocator out of space for allocations!\n");
				return;
			}

			slab->init();

			slab->attach_allocation(memory_for_blocks);

			local_slabs.init_claimed_index(slab, index);



		}



		__device__ void * malloc(){


			//add shebang here to calculate warp group
			storage_bitmap * team_storage = &storages[poggers::utils::get_smid()];


			while (true){

				int index = local_slabs.get_random_active_index();


				void * my_alloc = local_slabs.malloc(team_storage, index);

				if (my_alloc == nullptr){

					if (local_slabs.claim_index(index)){
						load_new_slab(index);
					}
					

				} else {
					return my_alloc;
				}

			}

		}


		//I will work on you someday my sweet summer child
		__device__ bool free (void * allocation){
			return true;
		}


		static __host__ my_type * generate_on_device(){

			auto dev_start = std::chrono::high_resolution_clock::now();

			my_type host_version;

			host_version.my_bitbuddy = bitbuddy_type::generate_on_device();

			//for now, this always positions itself on device 0.
			int num_sms = poggers::utils::get_num_streaming_multiprocessors(0);

			storage_bitmap * bitmaps;

			cudaMalloc((void **)&bitmaps, num_sms*sizeof(storage_bitmap));

			host_version.storages = bitmaps;


			my_type * dev_version;
			cudaMalloc((void **)&dev_version, sizeof(my_type));

			cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			//launch kernel to initialize

			init_simple_allocator<my_type><<<1,1024>>>(dev_version, num_sms);

			cudaDeviceSynchronize();


			auto dev_end = std::chrono::high_resolution_clock::now();


   			std::chrono::duration<double> dev_diff = dev_end-dev_start;


   
   			std::cout << "Constructed allocator in " << dev_diff.count() << " seconds\n";	


			return dev_version;

		}


		static __host__ void free_on_device(my_type * dev_version){


			my_type host_version;

			cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			cudaFree(host_version.storages);

			bitbuddy_type::free_on_device(host_version.my_bitbuddy);

			cudaFree(dev_version);
		}



	};
	

	//v2 of the simple allocator
	//uses a hash table to detect if an index should be freed.
	template <uint64_t alloc_size>
	struct simple_allocator_one_size_free {


		using my_type = simple_allocator_one_size<alloc_size>;

		using bitbuddy_type = bitbuddy_allocator<(1ULL << 12), 256>;

		//For now, lets reserve 4GB memory - make this template arg in full version
		//bytes desired = 4*1024*1024*1024;

		//2^32 in chunks of 2^14

		//2^18

		//TODO: make bitbuddy not use all the space lmao



		bitbuddy_type * my_bitbuddy;

		storage_bitmap * storages;

		//slab_retreiver * malloced_slabs;

		slab_storage<alloc_size> local_slabs;

		//set of helpful locks to stop bullshit
		uint64_t_bitarr locks;

		// __device__ void load_new_slab(){


		// 	alloc_bitarr * slab;

		// 	while (true){

		// 		slab = malloced_slabs->give_bitmap();

		// 		if (slab == nullptr){

		// 			//attempt to take lock
		// 			if (!(locks.set_index(0) & 1ULL)){

		// 				slab_retreiver * next_block = my_bitbuddy->malloc(sizeof(slab_retreiver));

		// 				void * memory_for_blocks = my_bitbuddy->malloc(alloc_size*4096*31);

		// 				if (next_block == nullptr || memory_for_blocks == nullptr){

		// 					printf("Buddy allocator out of space for allocations!\n");
		// 					return;
		// 				}



		// 			}	


		// 		}

		// 	}



		//Simple one size allocator - How to connect to slabs?
		__device__ void load_new_slab(int index){


			alloc_bitarr * slab = (alloc_bitarr *) my_bitbuddy->malloc(sizeof(alloc_bitarr));


			assert(((uint64_t) slab) % size(alloc_bitarr) == ((uint64_t) slab));

			void * memory_for_blocks = my_bitbuddy->malloc(alloc_size*4096);

			if (slab == nullptr || memory_for_blocks == nullptr){

				printf("Buddy allocator out of space for allocations!\n");
				return;
			}

			slab->init();

			slab->attach_allocation(memory_for_blocks);

			local_slabs.init_claimed_index(slab, index);



		}



		__device__ void * malloc(){


			//add shebang here to calculate warp group
			storage_bitmap * team_storage = &storages[poggers::utils::get_smid()];


			while (true){

				int index = local_slabs.get_random_active_index();


				void * my_alloc = local_slabs.malloc(team_storage, index);

				if (my_alloc == nullptr){

					if (local_slabs.claim_index(index)){
						load_new_slab(index);
					}
					

				} else {
					return my_alloc;
				}



			}

		}


		//I will work on you someday my sweet summer child
		__device__ bool free (void * allocation){
			return true;
		}


		static __host__ my_type * generate_on_device(){

			auto dev_start = std::chrono::high_resolution_clock::now();

			my_type host_version;

			host_version.my_bitbuddy = bitbuddy_type::generate_on_device();

			//for now, this always positions itself on device 0.
			int num_sms = poggers::utils::get_num_streaming_multiprocessors(0);

			storage_bitmap * bitmaps;

			cudaMalloc((void **)&bitmaps, num_sms*sizeof(storage_bitmap));

			host_version.storages = bitmaps;


			my_type * dev_version;
			cudaMalloc((void **)&dev_version, sizeof(my_type));

			cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			//launch kernel to initialize

			init_simple_allocator<my_type><<<1,1024>>>(dev_version, num_sms);

			cudaDeviceSynchronize();


			auto dev_end = std::chrono::high_resolution_clock::now();


   			std::chrono::duration<double> dev_diff = dev_end-dev_start;


   
   			std::cout << "Constructed allocator in " << dev_diff.count() << " seconds\n";	


			return dev_version;

		}


		static __host__ void free_on_device(my_type * dev_version){


			my_type host_version;

			cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			cudaFree(host_version.storages);

			bitbuddy_type::free_on_device(host_version.my_bitbuddy);

			cudaFree(dev_version);
		}



	};



}

}


#endif //GPU_BLOCK_