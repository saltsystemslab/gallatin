#ifndef POGGERS_TEMPLATE_BITBUDDY
#define POGGERS_TEMPLATE_BITBUDDY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/four_bit_bitarray.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;


#define LEVEL_CUTOF 0

#define PROG_CUTOFF 3


//Four bit version of the allocator
//this is *so* slow
//like seriously how did I think this was a good idea
//gonna try a pure VEB tree, should be a lot faster
//grouped allocs will just occur

//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 



template <int depth>
struct templated_bitbuddy_four{


	using my_type = templated_bitbuddy_four<depth>;

	enum { size = (1ULL << (4*depth)) };

	using child_type = templated_bitbuddy_four<depth-1>;


	static_assert(size > 0);
	
	
	four_bit_bitarray mask;

	child_type children[16];


	static __host__ my_type * generate_on_device(){



		my_type * dev_version;

		cudaMalloc((void **)& dev_version, sizeof(my_type));

		cudaMemset(dev_version, ~0U, sizeof(my_type));

		return dev_version;



	}

	static __host__ void free_on_device(my_type * dev_version){


		cudaFree(dev_version);

	}


	__host__ __device__ bool valid_for_alloc(uint64_t ext_size){

		return (size >= ext_size && ext_size > 15*child_type::size);

	}


	//grab a lock for a random full
	__device__ int acquire_random_lock(){


		while (true) {
			int index = mask.shrink_index(mask.get_random_all_avail());
		}

	}


	__device__ uint64_t malloc_at_level(){

		while (true){

			int index = mask.shrink_index(mask.get_random_all_avail());

			if (index == -1) return (~0ULL);

			//grab lock first
			uint64_t old = mask.unset_lock_bit_atomic(index);




			//someone else is working with this lock

			if (!(old & SET_LOCK_BIT_FOUR(index))){
				continue;
			}

			


			//3 cause we unlocked it
			if (__popcll(old & READ_ALL_FOUR(index)) == 3){


				mask.unset_all_atomic(index);

				return index * size;

			} else {
				mask.set_lock_bit_atomic(index);
			}


		}

	}

	__device__ uint64_t malloc_group(int num_contiguous){


		while (true){

			int index = mask.shrink_index(mask.get_first_x_contiguous(num_contiguous));

			if (index == -1) return (~0ULL);

			uint64_t lock_mask = mask.x_lock_mask(num_contiguous) << (4*index);

			uint64_t acquired_bits = mask.unset_bits(lock_mask) & lock_mask;

			uint64_t acquired_locks = acquired_bits & lock_mask;

			if (__popcll(acquired_locks) == num_contiguous){

				//success? we grabbed all locks
				acquired_bits = (acquired_bits) & (acquired_bits >> 1);

				acquired_bits = (acquired_bits) & (acquired_bits >> 1) & lock_mask;

				if (__popcll(acquired_bits) == num_contiguous){

					//true success
					//unset the bits

					//first, unset child bits
					mask.unset_bits(acquired_bits << 1);

					//then, unset team bits
					mask.unset_bits(acquired_bits << 2);

					//and unset the last bit
					mask.unset_continue_bit_atomic(index+num_contiguous-1);

					//free blocks are left in locked state

					return index*size;

				}

		}

		//failures float out and locks are deacquired
		mask.set_bits(acquired_locks);


	}

	}


	//cycle endlessly until you find an allocation or none are available.
	__device__ uint64_t malloc_child(uint64_t bytes_needed){

		while (true){


			//first, check if there is a preallocated child available.
			int index = mask.shrink_index(mask.get_first_child_only());

			if (index != -1){

				uint64_t offset = children[index].malloc_offset(bytes_needed);

				if (offset != (~0ULL)){
					return index*size+offset;
				} else {

					//if we failed to read from one of these, then it must not be available
					if (mask.unset_lock_bit_atomic(index) & SET_LOCK_BIT_FOUR(index)){

						//if we are in an understandable state.
						//if (mask & SET_CHILD_BIT_FOUR(offset)){
						mask.unset_child_bit_atomic(index);
						//}
						
						mask.set_lock_bit_atomic(index);
					}

				}

				//if we didn't get the alloc, retry?
				continue;

			}

			index = mask.shrink_index(mask.get_first_child_lock());

			if (index == -1) return (~0ULL);

			if(mask.unset_lock_bit_atomic(index) & SET_LOCK_BIT_FOUR(index)){

				//mask definitely has children available! lets set it to a valid state
				if (mask & SET_CHILD_BIT_FOUR(index)){

					//is this slow? yep - I'll fuse them later
					mask.unset_alloc_bit_atomic(index);
					mask.unset_continue_bit_atomic(index);
					mask.set_lock_bit_atomic(index);

					//uint64_t offset = children[index].malloc_offset(bytes_needed);

				}


			}


			mask.global_load_this();

		}

	}

	// __device__ uint64_t malloc_child_v3_temp(uint64_t bytes_needed){

	// 	while (true){

	// 		//mask.global_load_this();

	// 		int index = mask.shrink_index(mask.get_random_active_bit_control());

	// 		if (index == -1) return (~0ULL);


	// 		uint64_t offset = children[index].malloc_offset(bytes_needed);

	// 		if (offset == (~0ULL)){
	// 			mask.unset_control_bit_atomic(index);
	// 			continue;
	// 		}

	// 		if (mask & SET_FIRST_BIT(index)){

	// 			if (mask.unset_lock_bit_atomic() & SET_SECOND_BIT){

	// 			} else {

	// 				//00
	// 				children[index].free(offset);
	// 			}
	// 		}



	// 		if ((mask & SET_SECOND_BIT(index)) && (!(mask & SET_FIRST_BIT(index)))){

	// 			offset = children[index].malloc_offset(bytes_needed);

	// 		} else if (mask.unset_lock_bit_atomic(index) & SET_SECOND_BIT(index)){

	// 			offset = children[index].malloc_offset(bytes_needed);

	// 		} else {

	// 			mask.global_load_this();
	// 			continue;
	// 		}

	// 		if (offset == (~0ULL)){
	// 			mask.unset_control_bit_atomic(index);
	// 			continue;
	// 		}

	// 		return index*size+offset;

	// 	}

	// }


	__device__ uint64_t malloc_offset(uint64_t bytes_needed){


		uint64_t offset;

		if (valid_for_alloc(bytes_needed)){

			offset = malloc_at_level();

		} else {

			offset = malloc_child(bytes_needed);

		}


		return offset;



	}


	__device__ bool free_at_level(uint64_t offset){

		//at level we will force to be 0000
		//no one else can lock.
		if ((mask.set_all_atomic(offset) & READ_ALL_FOUR(offset)) == 0){

			return true;
		}

		return false;

	}

	__device__ bool free(uint64_t offset){

		uint64_t local_offset = offset / size;

		assert(local_offset < 32);

		if (children[local_offset].free(offset % size)){

			mask.unset_lock_bit_atomic(local_offset);
			if (children[local_offset].all_free()){

				//reset on full reallocation
				mask.set_all_atomic(local_offset);

			} else {

				mask.set_child_bit_atomic(local_offset);

			}

			return true;

		}

		return free_at_level(local_offset);

	}





};


template <>
struct  templated_bitbuddy_four<0> {

	using my_type = templated_bitbuddy_four<0>;

	enum {size = 1};
	//TODO double check this, feels like it should be sizeinbytes/32;

	four_bit_bitarray mask;


	__device__ uint64_t malloc_offset(uint64_t bytes_needed){

		return malloc_at_level();
	}


	__device__ uint64_t malloc_group(int num_contiguous){


			while (true){

				int index = mask.shrink_index(mask.get_first_x_contiguous(num_contiguous));

				if (index == -1) return (~0ULL);

				uint64_t lock_mask = mask.x_lock_mask(num_contiguous) << (4*index);

				uint64_t acquired_bits = mask.unset_bits(lock_mask) & lock_mask;

				uint64_t acquired_locks = acquired_bits & lock_mask;

				if (__popcll(acquired_locks) == num_contiguous){

					//success? we grabbed all locks
					acquired_bits = (acquired_bits) & (acquired_bits >> 1);

					acquired_bits = (acquired_bits) & (acquired_bits >> 1) & lock_mask;

					if (__popcll(acquired_bits) == num_contiguous){

						//true success
						//unset the bits

						//first, unset child bits
						mask.unset_bits(acquired_bits << 1);

						//then, unset team bits
						mask.unset_bits(acquired_bits << 2);

						//and unset the last bit
						mask.unset_continue_bit_atomic(index+num_contiguous-1);

						//free blocks are left in locked state

						return index*size;

					}

			}

			//failures float out and locks are deacquired
			mask.set_bits(acquired_locks);


		}

	}

	__device__ uint64_t malloc_at_level(){

		while (true){

			int index = mask.shrink_index(mask.get_random_all_avail());

			if (index == -1) return (~0ULL);

			//grab lock first
			uint64_t old = mask.unset_lock_bit_atomic(index);




			//someone else is working with this lock

			if (!(old & SET_LOCK_BIT_FOUR(index))){
				continue;
			}

			


			//4 cause we unlocked it but not in the old version we see
			if (__popcll(old & READ_ALL_FOUR(index)) == 4){


				mask.unset_all_atomic(index);

				return index * size;

			} else {
				mask.set_lock_bit_atomic(index);
			}


		}

	}



	//returns true if entirely full
	__device__ bool free_at_level(uint64_t offset){


		if (__popcll(mask.set_all_atomic(offset) & READ_ALL_FOUR(offset)) == 0){

			return true;
		}

		return false;

	}

	__device__ bool free(uint64_t offset){

		return free_at_level(offset);

	}

	__host__ __device__ bool valid_for_alloc(uint64_t size){
		return true;
	}


	__host__ __device__ bool all_free(){



		uint64_t lock_mask = mask.x_lock_mask(16);
		uint64_t old = mask.unset_bits(lock_mask);

		if (__popcll(old & lock_mask) == 16){

			if ((~mask) == 0){
				return true;
			}
		}

		//return locks we grabbed
		mask.set_bits(old & lock_mask);
		return false;

	}

	//only occurs when we grabbed *all* the locks
	__host__ __device__  void return_all_free(){

		mask.set_bits(mask.x_lock_mask(16));
	}


	static __host__ my_type * generate_on_device(){



		my_type * dev_version;

		cudaMalloc((void **)& dev_version, sizeof(my_type));

		cudaMemset(dev_version, ~0U, sizeof(my_type));

		return dev_version;



	}

	static __host__ void free_on_device(my_type * dev_version){


		cudaFree(dev_version);

	}

};


//32 should return 0

template <uint64_t num_allocations>
struct determine_depth {


	enum {depth = num_allocations > 32 ? determine_depth<num_allocations/32>::depth+1 : 0};


};


template <>
struct determine_depth<0> {


	enum {depth = 32};


};

template <int depth>
struct determine_num_allocations {

	enum {count = 32 * determine_num_allocations<depth-1>::count };

};

template <>
struct determine_num_allocations<0> {

	enum {count = 32};
};


// template <uint64_t num_allocations>
// struct bitbuddy_allocator {

// 	using my_type = bitbuddy_allocator<num_allocations>;


// 	using bitbuddy_type = templated_bitbuddy_four<determine_depth<num_allocations>::depth, determine_num_allocations<determine_depth<num_allocations>::depth>::count>;

	
// 	bitbuddy_type * internal_allocator;

// 	void * memory;


// 	static __host__ my_type * generate_on_device(){

// 		my_type host_version;

// 		void * ext_memory;

// 		host_version.internal_allocator = bitbuddy_type::generate_on_device();

// 		if (host_version.internal_allocator == nullptr){

// 			printf("Allocator could not get enough space\n");
// 			assert(1==0);
// 		}

// 		cudaMalloc((void **)&ext_memory, num_allocations*size_of_allocation);

// 		if (ext_memory == nullptr){

// 			cudaFree(host_version.internal_allocator);

// 			printf("Allocator could not get enough memory to handle requested # allocations.\n");
// 			assert(1==0);
// 		}

// 		host_version.memory = ext_memory;

// 		my_type * dev_version;

// 		//my type is 16 bytes. I'm gonna conservatively estimate that this will always go through.
// 		cudaMalloc((void **)&dev_version, sizeof(my_type));

// 		cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

// 		return dev_version;


// 	}

// 	static __host__ void free_on_device(my_type * dev_version){

// 		my_type host_version;

// 		cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

// 		cudaFree(dev_version);

// 		cudaFree(host_version.memory);

// 		bitbuddy_type::free_on_device(host_version.internal_allocator);

// 	}


// 	__device__ void * malloc(uint64_t bytes_needed){

// 		uint64_t offset = internal_allocator->malloc_offset((bytes_needed-1)/size_of_allocation+1);

// 		if (offset == (~0ULL)) return nullptr;

// 		return (void *) ((uint64_t) memory + offset*size_of_allocation); 
// 	}


// 	__device__ bool free(void * allocation){

// 		uint64_t offset = ((uint64_t) allocation - (uint64_t) memory)/size_of_allocation;

// 		return internal_allocator->free(offset);
// 	}


// 	__host__ __device__ bool is_bitbuddy_alloc(void * allocation){


// 		//all allocations from the bitbuddy must be in the stride of the bitbuddy.
// 		//since internals are offset by light of their allocation this is an easy check.
// 		return ((uint64_t) allocation % bitbuddy_type::lowest_size == allocation);
		
// 		//could also make sure offset is internal but whatevs.
// 		//kind of on you to not free from the wrong allocator lol
// 		//uint64_t offset = ((uint64_t allocation) - (uint64_t) memory);

// 	}

// };


}

}


#endif //GPU_BLOCK_