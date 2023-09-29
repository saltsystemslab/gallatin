#ifndef GALLATIN_CHAINING_HASH
#define GALLATIN_CHAINING_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>

#include <gallatin/data_structs/callocable.cuh>




namespace gallatin {

namespace data_structs {


	template <typename ht_type>
	__global__ void chaining_table_fill_buffers(ht_type * table){


		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= table->nblocks) return;


		auto block_ptr = &table->pointer_list[tid];

		table->attach_block(block_ptr);

	}


	template <typename ht_type>
	__global__ void calculate_chain_kernel(ht_type * table, uint64_t * max, uint64_t * avg, uint64_t nblocks){


		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nblocks) return;

		table->calculate_chain_length(max, avg, tid);

	}



	template <typename Key, typename Val, int size>
	struct chaining_block {

		using filled_type = chaining_block<Key, Val, size>;
		filled_type * next;

		Key keys[size];
		Val vals[size];


		__device__ void init(Key & defaultKey){

			//next points to nullptr
			atomicExch((unsigned long long int *)&next, 0ULL);

			for (int i = 0; i < size; i++){

				typed_atomic_exchange(&keys[i], defaultKey);

			}
		}

		__device__ bool insert(Key & defaultKey, Key insertKey, Val insertVal){

			for (int i = 0; i < size; i++){

				if (keys[i] == defaultKey){
					if (typed_atomic_write(&keys[i], defaultKey, insertKey)){
						vals[i] = insertVal;
						return true;
					}
					//return true;
				}
			}

			return false;

		}

		__device__ bool query(Key queryKey, Val & returnVal){


			for (int i =0; i < size; i++){


				if (keys[i] == queryKey){

					returnVal = vals[i];
					return true;
				}
			}

			return false;
		}

	};

	template <typename Key, typename Val, int size=4>
	struct chaining_table{

		using my_type = chaining_table<Key, Val, size>;

		uint64_t nslots;

		uint64_t nblocks;

		uint64_t seed;

		using block_type = chaining_block<Key, Val, size>;

		block_type ** pointer_list;

		//todo: make const for improved performance.
		Key defaultKey;

		static __host__ my_type * generate_on_device(uint64_t ext_nslots, Key ext_defaultKey, uint64_t ext_seed){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->seed = ext_seed;
			host_version->nslots = ext_nslots;

			block_type ** ext_pointer_list;

			host_version->nblocks = ext_nslots/size;


			host_version->defaultKey = ext_defaultKey;

			cudaMalloc((void **)&ext_pointer_list, host_version->nblocks*sizeof(block_type *));

			//set all slots to nullptr.
			cudaMemset(ext_pointer_list, 0, host_version->nblocks*sizeof(block_type *));

			host_version->pointer_list = ext_pointer_list;

			return gallatin::utils::move_to_device(host_version);


		}

		static __host__ my_type * generate_on_device_prealloc(uint64_t ext_nslots, Key ext_defaultKey, uint64_t ext_seed){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->seed = ext_seed;
			host_version->nslots = ext_nslots;

			block_type ** ext_pointer_list;

			host_version->nblocks = ext_nslots/size;

			host_version->defaultKey = ext_defaultKey;

			cudaMalloc((void **)&ext_pointer_list, host_version->nblocks*sizeof(block_type *));

			//set all slots to nullptr.
			cudaMemset(ext_pointer_list, 0, host_version->nblocks*sizeof(block_type *));

			host_version->pointer_list = ext_pointer_list;

			my_type * device_version =  gallatin::utils::move_to_device(host_version);

			chaining_table_fill_buffers<my_type><<<(ext_nslots/size-1)/256+1,256>>>(device_version);

			cudaDeviceSynchronize();

			return device_version;


		}

		static __host__ void free_on_device(my_type * device_version){

			auto host_version = gallatin::utils::move_to_host(device_version);

			cudaFree(host_version->pointer_list);

			cudaFreeHost(host_version);

		}


		//format new block and attempt to atomicCAS
		__device__ bool attach_block(block_type ** block_ptr){

			block_type * new_block = (block_type *) gallatin::allocators::global_malloc(sizeof(block_type));


			if (new_block == nullptr) return false;

			new_block->init(defaultKey);

			if (atomicCAS((unsigned long long int *)block_ptr, 0ULL, (unsigned long long int ) new_block) != 0ULL){

				gallatin::allocators::global_free(new_block);

			}

			return true;

		}


		__device__ void calculate_chain_length(uint64_t * max_len, uint64_t * avg_len, uint64_t my_index){

			uint64_t my_length = 0;

			block_type * my_block = pointer_list[my_index];

			while (my_block != nullptr){
				my_length += 1;
				my_block = my_block->next;
			}

			atomicMax((unsigned long long int *)max_len, (unsigned long long int) my_length);
			atomicAdd((unsigned long long int *)avg_len, (unsigned long long int) my_length);

		}

		__device__ void insert(Key newKey, Val newVal){


			uint64_t my_slot = gallatin::hashers::MurmurHash64A(&newKey, sizeof(Key), seed) % nblocks;

			//attempt to read my slot

			block_type * my_block = pointer_list[my_slot];

			block_type ** my_pointer_addr = &pointer_list[my_slot];

			while (true){


				if (my_block == nullptr){
					//failure to find new segment
					if (!attach_block(my_pointer_addr)) return;

					my_block = my_pointer_addr[0];
					continue;
				}

				//otherwise, try to insert

				if (my_block->insert(defaultKey, newKey, newVal)){
					return;
				}

				//otherwise, move to next block.

				my_pointer_addr = &my_block->next;
				my_block = my_block->next;



			}



			return;
		}

		__device__ bool query(Key queryKey, Val & returnVal){

			uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;

			block_type * my_block = pointer_list[my_slot];

			//block_type ** my_pointer_addr = &pointer_list[my_slot];

			while (my_block != nullptr){

				if (my_block->query(queryKey, returnVal)){
					return true;
				}

				my_block = my_block->next;
			}

			return false;

		}



		__host__ void print_chain_stats(){

			my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

			uint64_t nblocks = host_version->nblocks;

			uint64_t * max;
			uint64_t * avg;

			cudaMallocManaged((void ** )&max, sizeof(uint64_t));
			cudaMallocManaged((void **)&avg, sizeof(uint64_t));

			max[0] = 0;
			avg[0] = 0;

			calculate_chain_kernel<my_type><<<(nblocks-1)/256+1, 256>>>(this, max, avg,nblocks);

			cudaDeviceSynchronize();

			std::cout << "Chains - Max: " << max[0] << ", Avg: " << 1.0*avg[0]/nblocks << ", nblocks: " << nblocks << std::endl;

			cudaFree(max);
			cudaFree(avg);
		}

	};


}


}


#endif //end of resizing_hash guard