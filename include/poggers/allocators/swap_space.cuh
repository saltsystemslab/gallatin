#ifndef SWAP_SPACE
#define SWAP_SPACE


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


	class serializable {

		public:
	}


	struct object;
	template <typename Referent>;
	struct pointer;

	template <typename Referent>;
	struct pin;



	//The swap space is a silent manager of cuda/cpp pointers
	//The concept is taken from the original implementation of the b-epsilon tree
	//which uses a swap space to automatically pin and depin memory in RAM,
	//allowing the data structure to efficiently scale to disk. 


	//The goal of this code is to abstract the transition from host/device code for CUDA
	// data structures, such that uses can write complex data structs and have them scale automatically to disk.
	//As per the original implementation, we define a swap_space::pointer type that acts a device pointer.

	template <typename dev_allocator, typename host_allocator>
	struct swap_space {


		dev_allocator * dev_memory;

		host_allocator * host_memory;


		uint64_t max_in_memory_objects;
		uint64_t current_in_memory_objects= 0;

		//need queue type and hash table type


		using hash_table_type = 



		using my_type = swap_space<dev_allocator, host_allocator>;

		__host__ my_type * generate_on_device(dev_allocator * dev_alloc, host_allocator * host_alloc){

			my_type * host_version;

			cudaMallocHost((void **)&host_version, sizeof(my_type));

			host_version->dev_memory = dev_allocator;
			host_allocator->host_memory = host_allocator;

			my_type * dev_version;

			cudaMalloc((void **)&dev_version, sizeof(my_type));

			cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

			cudaFreeHost(host_version);

			return dev_version;

		}

		//does not clean up allocators it is based on.
		//they are assumed to be their own objects
		__host__ void free_on_device(my_type * dev_version){

			cudaFree(dev_version);

		}
		//convert a raw cuda pointer to a new host pointer, and return the host pointer
		template <typename object_type>
		__device__ void serialize(object_type * dev_pointer){

			object_type * host_ptr = (object_type *) host_memory->malloc(sizeof(object_type));

			zero_copy_dev_host(dev_pointer, host_ptr);
		}

		//convert a host pointer to a cuda dev pointer.
		template <typename object_type>
		__device__ void deserialize(object_type * host_ptr){

			object_type * dev_pointer = (object_type *) dev_memory->malloc(sizeof(object_type));

			zero_copy_host_dev(host_ptr, dev_pointer);

		}

		template <class Referent>
		pointer<Referent> allocate(Referent * tgt){
			return pointer<Referent>(this, tgt);
		}


		//pinned memory
		//For the lifetime of a pin the pinned object is in GPU memory.
		template <class Referent>
		struct pin {

			swap_space *ss;
        	uint64_t target;

			const Referent * operator->(void) const {

				#if DEBUG_ASSERTS
				assert(ss->objects.count(target) > 0)
				#endif

			}
		};


		template <class Referent>
		struct pointer : public serializable {

		}




		struct object {

			object(swap_space *sspace, serializable *tgt){



				id = sspace->get_unique_id();

				count = 0;

			}

			serializable * host_target;
			serializable * dev_target;
			uint64_t id;
			uint64_t count;

		};
		




	};


}

}


#endif //GPU_BLOCK_