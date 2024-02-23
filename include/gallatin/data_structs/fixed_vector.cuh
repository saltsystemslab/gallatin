#ifndef GALLATIN_FIXED_VECTOR
#define GALLATIN_FIXED_VECTOR


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


namespace gallatin {

namespace data_structs {

	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current


	template <typename vector> 
	__global__ void init_dev_vector(vector * vec){


	uint64_t tid = gallatin::utils::get_tid();

	if (tid != 0) return;

	vec->init();

	}

	template <typename vector> 
	__global__ void free_dev_vector(vector * vec){


	uint64_t tid = gallatin::utils::get_tid();

	if (tid != 0 )return;

	vec->free_vector();

	}

	template <typename T, typename vector>
	__global__ void copy_vector_to_dev_array(T * dev_array, vector * dev_vector, uint64_t size){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= size) return;

		dev_array[tid] = dev_vector[0][tid];


	}

	//Vector - a resizable templated container built for high performance in CUDA.
	
	//Features:
	//  -- configurable typing in a header-only format for fast execution
	//  -- additional workers are automatically pulled into the environment to assist resizes.
	//  == TODO: allow for dynamic parallelism if #workers crosses lower threshold?


	//T - type inserted and retreived from the vector
	//Allocator - allocator type - this is Gallatin normally, but let's expose it to other systems.
	//resizable - allow the vector to be resized 
	// - if not true insertions can fail instead of expanding the vector
	//    but space usage will not increase past the set size.
	template <typename T, uint64_t min_items, uint64_t max_items, bool on_host=false>
	struct fixed_vector {

		using my_type = fixed_vector<T, min_items, max_items, on_host>;

		static const uint n_directory = (gallatin::utils::numberOfBits(max_items-1)+1-gallatin::utils::numberOfBits(min_items-1));

		static const uint min_bits = gallatin::utils::numberOfBits(min_items-1)+1;

		static const uint64_t nbits = 2*max_items; 

		//directory is the set of live arrays.
		T * directory[(gallatin::utils::numberOfBits(max_items-1)-gallatin::utils::numberOfBits(min_items-1)+1)];

		uint64_t locks;

		uint64_t size;


		__device__ bool add_new_backing(uint64_t directory_index){


			if (atomicOr((unsigned long long int *)&locks, (unsigned long long int)SET_BIT_MASK(directory_index)) & SET_BIT_MASK(directory_index)) return false;


			uint64_t new_size = min_items << (directory_index-1);
			if (directory_index <= 1){
				new_size = min_items;
			}

			//printf("Starting malloc\n");
			T * new_backing = (T *) gallatin::allocators::global_malloc_combined((sizeof(T)*new_size), on_host);

			//printf("Ending malloc\n");
			//atomicExch breaks here?
			atomicCAS((unsigned long long int *)&directory[directory_index], (unsigned long long int)0ULL, (unsigned long long int) new_backing);

			return true;
		}

		__device__ void init(){

			size = 0;

			locks = 0;

			for (uint64_t i = 0; i < n_directory; i++){
				directory[i] = nullptr;
			}

			__threadfence();

			add_new_backing(0);

			return;

		}

		__device__ fixed_vector(){

			init();

		}

		static __host__ my_type * get_device_vector(){

			my_type * dev_version = gallatin::utils::get_device_version<my_type>();

			init_dev_vector<<<1,1>>>(dev_version);

			return dev_version;

		}

		//delete memory used by this vector.
		//atm not really threadsafe so only do it if you're sure.
		__device__ void free_vector(){


			for (uint64_t i = 0; i < gallatin::utils::__cfcll(locks); i++){

				gallatin::allocators::global_free_combined(directory[i], on_host);

			}

			return;

		}


		__host__ static void free_device_vector(my_type * dev_version){

			free_dev_vector<<<1,1>>>(dev_version);

		}

		__device__ uint64_t get_directory_index(uint64_t item_index){

			uint64_t index = 0;

			uint64_t items_covered = min_items;

			while (true){

				if (item_index < items_covered){

					return index;

				}

				index+=1;

				//if (index == 1) items_covered = min_items;
				items_covered = items_covered << 1;

			}



		}


		__device__ uint64_t get_local_position(uint64_t clipped_hash, uint64_t index){

			if (index == 0) return clipped_hash;

			if (index == 1) return clipped_hash - min_items;

			uint64_t items_at_level_below = min_items + min_items << (index-2);

			return clipped_hash - items_at_level_below;


		}



		//reuturns the index written.
		__device__ uint64_t insert(T item){


			uint64_t my_index = atomicAdd((unsigned long long int *)&size, 1ULL);

			//too big!
			if (my_index >= max_items) return ~0ULL;

			uint64_t directory_index = get_directory_index(my_index);

			if (directory_index >= n_directory){

				//uint64_t alt_directory_index = get_directory_index(my_index);
				//printf("Index %lu has directory %lu > %lu\n", my_index, directory_index, n_directory);
				return ~0ULL;
			}

			uint64_t local_index = get_local_position(my_index, directory_index);


			T * global_read_directory = directory[directory_index];

			if (global_read_directory == nullptr){

				add_new_backing(directory_index);


				while (global_read_directory == nullptr){
						global_read_directory = (T *) gallatin::utils::ld_acq((uint64_t *)&directory[directory_index]);
				}

			}

			global_read_directory[local_index] = item;

			return my_index;

		}



		__device__ uint64_t bulk_insert(cg::coalesced_group & active_threads, T item){


		    uint64_t my_group_sum = 1;

		    my_group_sum = cg::exclusive_scan(active_threads, my_group_sum, cg::plus<uint64_t>());

		    //last thread in group has total size and controls atomic

		    uint64_t old_count;

		    if (active_threads.thread_rank() == active_threads.size()-1){

		      old_count = atomicAdd((unsigned long long int *)&size, my_group_sum+1);

		    }

    		old_count = active_threads.shfl(old_count, active_threads.size()-1);

    		uint64_t my_index = old_count + my_group_sum;


			//uint64_t my_index = atomicAdd((unsigned long long int *)&size, 1ULL);

			//too big!
			if (my_index >= max_items) return ~0ULL;

			uint64_t directory_index = get_directory_index(my_index);

			if (directory_index >= n_directory){

				//uint64_t alt_directory_index = get_directory_index(my_index);
				//printf("Index %lu has directory %lu > %lu\n", my_index, directory_index, n_directory);
				return ~0ULL;
			}

			uint64_t local_index = get_local_position(my_index, directory_index);


			T * global_read_directory = directory[directory_index];

			if (global_read_directory == nullptr){

				add_new_backing(directory_index);


				while (global_read_directory == nullptr){
						global_read_directory = (T *) gallatin::utils::ld_acq((uint64_t *)&directory[directory_index]);
				}

			}

			global_read_directory[local_index] = item;

			return my_index;

		}

		//deference operator - double check that memory has been cleared before giving back.
		__device__ T& operator[](uint64_t my_index)
		{

			uint64_t directory_index = get_directory_index(my_index);

			uint64_t local_index = get_local_position(my_index, directory_index);

		    

		   return directory[directory_index][local_index];
		   
		}


		// static __host__ std::vector<T> export_to_host_device(my_type * device_version){

		// 	std::vector<T> host_vector;

		// 	my_type * host_version = gallatin::utils::copy_to_host(device_version);

		// 	uint64_t size = host_version->size;

		// 	uint64_t min_size = 0;

		// 	uint64_t max_size = min_items;

		// 	uint64_t directory_index = 0;

		// 	while (size < max_size){

		// 		uint64_t n_items = max_size-min_size;

		// 		if (host_version->directory[directory_index] == nullptr){

		// 			cudaFreeHost(host_version);

		// 			return host_vector;
		// 		}

		// 		T * host_array = gallatin::utils::copy_to_host<T>(host_version->directory[directory_index], n_items);

		// 		for (uint64_t i = 0; i < n_items; i++){

		// 			host_vector.push_back(host_array[i]);

		// 		}

		// 		cudaFreeHost(host_array);

		// 		min_size = max_size;

		// 		max_size *= 2;

		// 		directory_index+=1;

		// 	}

		// 	//last pass
		// 	if (host_version->directory[directory_index] == nullptr){

		// 			cudaFreeHost(host_version);

		// 			return host_vector;
		// 		}

		// 	uint64_t final_n_items = size-min_size;

		// 	T * host_array = gallatin::utils::copy_to_host<T>(host_version->directory[directory_index], final_n_items);

		// 	for (uint64_t i = 0; i < final_n_items; i++){
		// 		host_vector.push_back(host_array[i]);
		// 	}

		// 	cudaFreeHost(host_array);


		// 	//final cleanup
		// 	cudaFreeHost(host_version);

		// 	return host_vector;

		// }	

		// //host variant
		// static __host__ std::vector<T> export_to_host_host(my_type * device_version){

		// 	std::vector<T> host_vector;

		// 	my_type * host_version = gallatin::utils::copy_to_host(device_version);

		// 	uint64_t size = host_version->size;

		// 	uint64_t min_size = 0;

		// 	uint64_t max_size = min_items;

		// 	uint64_t directory_index = 0;

		// 	while (size < max_size){

		// 		uint64_t n_items = max_size-min_size;

		// 		if (host_version->directory[directory_index] == nullptr){

		// 			cudaFreeHost(host_version);

		// 			return host_vector;
		// 		}

		// 		T * host_array = host_version->directory[directory_index];

		// 		for (uint64_t i = 0; i < n_items; i++){

		// 			host_vector.push_back(host_array[i]);
					
		// 			if (host_array[i] == 0){
		// 				printf("BUG\n");
		// 			}

		// 		}


		// 		min_size = max_size;

		// 		max_size *= 2;

		// 		directory_index+=1;

		// 	}

		// 	//last pass
		// 	if (host_version->directory[directory_index] == nullptr){

		// 			cudaFreeHost(host_version);

		// 			return host_vector;
		// 		}

		// 	uint64_t final_n_items = size-min_size;

		// 	T * host_array = host_version->directory[directory_index];

		// 	for (uint64_t i = 0; i < final_n_items; i++){

		// 		if (host_array[i] == 0){
		// 			printf("BUG\n");
		// 		}
		// 		host_vector.push_back(host_array[i]);
		// 	}


		// 	//final cleanup
		// 	cudaFreeHost(host_version);

		// 	return host_vector;





		// }

		static __host__ std::vector<T> export_to_host(my_type * device_version){


			



			my_type * host_version = gallatin::utils::copy_to_host(device_version);

			uint64_t size = host_version->size;

			std::vector<T> host_vector(size);

			//printf("Host vector has %lu items, %lu bytes\n", size, sizeof(T)*size);

			//T * host_array = (T *) malloc(sizeof(T)*size);

			T * device_array;

			GPUErrorCheck(cudaMalloc((void **)&device_array, sizeof(T)*size));

			copy_vector_to_dev_array<T, my_type><<<(size-1)/256+1, 256>>>(device_array, device_version, size);

			cudaDeviceSynchronize();

			GPUErrorCheck(cudaMemcpy(host_vector.data(), device_array, sizeof(T)*size, cudaMemcpyDeviceToHost));

			GPUErrorCheck(cudaDeviceSynchronize());

			cudaFree(device_array);

			//printf("Size of vector: %lu\n", host_vector.size());

			return host_vector;


		}



	};


}


}


#endif //end of queue name guard