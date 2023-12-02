#ifndef GALLATIN_EXTENDIBLE_HASH
#define GALLATIN_EXTENDIBLE_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>

#include <gallatin/data_structs/callocable.cuh>

//#include <gallatin/data_structs/formattable.cuh>

#include <gallatin/data_structs/formattable_atomics_recursive.cuh>


#define USE_ATOMICS 1

//atomic version of the table
//ldcg is for chumps...
//this is to verify correctness before moving to looser instructions.
namespace gallatin {

namespace data_structs {


	using namespace gallatin::allocators;
	using namespace gallatin::utils;


	template <typename T>
	__global__ void init_expanding_pointer_array(T * pointer_array){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid != 0) return;

		pointer_array->add_new_backing(0);

	}


	template <typename T, uint64_t min_items, uint64_t max_items>
	struct expanding_pointer_array {


		using my_type = expanding_pointer_array<T, min_items, max_items>;

		
		static const uint n_directory = (gallatin::utils::numberOfBits(max_items)-gallatin::utils::numberOfBits(min_items)+1);

		static const uint min_bits = gallatin::utils::numberOfBits(min_items-1);

		static const uint64_t nbits = 2*max_items; 

		uint64_t level;
		T ** directory[(gallatin::utils::numberOfBits(max_items)-gallatin::utils::numberOfBits(min_items)+1)];
		//uint64_t * live_bits;


		__device__ bool add_new_backing(uint64_t expected_backing){

			//needs to check that size is what I expect
			uint64_t my_addr = atomicCAS((unsigned long long int *)&level, (unsigned long long int) expected_backing, (unsigned long long int) expected_backing+1);


			if (my_addr != expected_backing) return false;

			uint64_t my_size = min_items;


			//first two slots are min size
			if (my_addr > 1){

				my_size = min_items << (my_addr-1);

			}


			T **  new_pointer = (T**) gallatin::allocators::global_malloc((sizeof(T *))*my_size);

			atomicExch((unsigned long long int *)&directory[my_addr], (unsigned long long int )new_pointer);

			//pointer set so done.

			return true;


		}


		static __host__ my_type * generate_on_device(){



			my_type * host_version = gallatin::utils::get_host_version<my_type>();



			//host_version[0] = default_host_version;

			//host_version->live_bits = gallatin::utils::get_device_version<uint64_t>(host_version->nbits);

			host_version->level = 0;


			//printf("Live bits %llu, max items: %llu\n", host_version->nbits, max_items);

			//cudaMemset(host_version->live_bits, 0ULL, sizeof(uint64_t)*max_items*2);


			my_type * device_version = gallatin::utils::move_to_device(host_version);

			init_expanding_pointer_array<my_type><<<1,1>>>(device_version);

			return device_version;


		}


		__device__ bool is_bucket_live(uint64_t bucket){

			// uint64_t high = bucket/64;

			// uint64_t low = bucket % 64;

			auto address = get_bucket_address(bucket)[0];

			//printf("Address: %llx\n", (uint64_t) address);

			return (address != nullptr);

		}

		//generate the masked hash for a given level
		//can be applied iteratively as long as level is monotonically decreasing.
		__device__ uint64_t mask_to_level(uint64_t hash, uint level){
		

			//printf("Min bits: %d\n", min_bits);

			uint64_t output = (hash & ((1ULL << (level+min_bits)) -1));


			printf("Input is %llu, output is %llu\n", hash, output);
			return output;

		}


		//intermediate slot is actual position in array
		//given level + index, start clipping
		//this overwrite old values stored in level + index.
		__device__ void determine_intermediate_slot(uint64_t & level, uint64_t & index){


			while (true){

				if (level == 1) return; 

				uint64_t size_of_level_below = min_items << (level-1);


				printf("Level %llu, index %llu, size_of_level_below %llu\n", level, size_of_level_below, index);

				if (index >= size_of_level_below){

					index = index - size_of_level_below;
					return;

				}

				//otherwise this points to a bucket in a lower level.	
				level -=1;


			}

			printf("Level is %llu, index is %llu\n", level, index);



		}

		__device__ T ** get_live_bucket_address(uint64_t index){


			uint64_t global_level = gallatin::utils::ldcv(&level);

			//while global_level >= 0
			while (true){


				index = mask_to_level(index, global_level);

				if (is_bucket_live(index)){

					determine_intermediate_slot(global_level, index);

					return &directory[global_level-1][index];
				}



				global_level-=1;

			}

		}

		__device__ T ** get_bucket_address(uint64_t index){


			uint64_t global_level = gallatin::utils::ldcv(&level);


			index = mask_to_level(index, global_level-1);


			determine_intermediate_slot(global_level, index);

			printf("Output of intermediate - level %llu, index %llu\n", global_level, index);

			return &directory[global_level-1][index];


		}

		//return pointer to memory - attempt to use lazy load!
		//same technique as calloced memory.
		//if pointer hasn't been set 
		__device__ T * get_bucket_pointer(uint64_t index){


			T ** address = get_live_bucket_address(index);

			return address[0];



		}


		__device__ void set_bucket_pointer(uint64_t index, T * new_bucket){

			T ** address = get_bucket_address(index);


			uint64_t result = atomicCAS((unsigned long long int *)address, 0ULL, (unsigned long long int) new_bucket);

			if (result != 0ULL){
				printf("Failed to attach %llu: result is %llx\n", index, result);
			}

			uint64_t high = index/64;
			uint64_t low = index % 64;

			//atomicOr((unsigned long long int *)&live_bits[high], SET_BIT_MASK(low));


		}
	
	};


	template <typename Key, typename Val>
	struct extendible_key_val_pair {

		Key key;
		Val val;

	};

	//block type for extendible hashing

	//pipeline
	//on size promotion up size.
	//this signals to incoming threads that the resize may occur now
	//and alternate slots may need to be queried.
	//success in promotion indicates control
	//you are the arbiter of setting the new bucket
	//this means atomicExch on old size and setting live bit.
	template <typename Key, Key defaultKey, typename Val, int num_pairs>
	struct extendible_bucket {

		using my_type = extendible_bucket<Key, defaultKey, Val, num_pairs>;


		//determine sizing
		//16 bits always reserved for size.
		uint16_t size;

		uint16_t padding[3];
		//do metadataBS?
		extendible_key_val_pair<Key, Val> slots [num_pairs];





		__device__ bool insert(Key ext_key, Val ext_val, uint16_t expected_size, uint16_t & internal_read_size){


			//first read size
			internal_read_size = gallatin::utils::ldcv(&size);

			//failure means resize has started...
			if (internal_read_size != expected_size) return false;


			//otherwise attempt insert

			for (int i = 0; i < num_pairs; i++){

				if (slots[i].key == defaultKey){

					//attempt update!

					if (typed_atomic_write(&slots[i].key, defaultKey, ext_key)){

						typed_atomic_exchange(&slots[i].val, ext_val);

						return true;

					}

				}

			}


			//all slots occupado
			return false;



		}

		__device__ bool query(Key ext_key, Val & ext_val, uint16_t & internal_read_size){

			//asserts that query may nnot be in another bucket.
			internal_read_size = gallatin::utils::ldcv(&size);


			for (int i = 0; i < num_pairs; i++){


				if (slots[i].key == ext_key){

					ext_val = gallatin::utils::ldcv(&slots[i].val);
					return true;

				}

			}

			return false;

		}

		//returns expected size if promotion is successful
		//if this fails someone else is in charge of promoting.

		__device__ uint16_t promote_size(uint16_t expected_size){

			return atomicCAS((unsigned short int *)&size, (unsigned short int) expected_size, (unsigned short int) expected_size+1);

		}


	};

	template <typename T>
	__global__ void init_exp_hash_table(T * dev_table, uint64_t min_size){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= min_size) return;

		printf("Attaching %llu\n", tid);

		dev_table->attach_new_bucket(tid);


	}

	template <typename Key, Key defaultKey, typename Val, int num_slots, uint64_t min_size, uint64_t max_size>
	struct extendible_hash_table {


		using node_type = extendible_bucket<Key, defaultKey, Val, num_slots>;

		using backing_type = expanding_pointer_array<node_type, min_size, max_size>;


		using my_type = extendible_hash_table<Key, defaultKey, Val, num_slots, min_size, max_size>;

		backing_type directory;


		static __host__ my_type * generate_on_device(){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			backing_type * ext_backing_pointer = backing_type::generate_on_device();

			//cursed but ehh


			backing_type * host_backing = gallatin::utils::move_to_host(ext_backing_pointer);


			cudaMemcpy(&host_version->directory, host_backing, sizeof(backing_type), cudaMemcpyHostToHost);

			//host_version->directory = host_backing;

			cudaFreeHost(host_backing);


			my_type * device_version = gallatin::utils::move_to_device(host_version);


			init_exp_hash_table<<<(min_size-1)/256+1,256>>>(device_version, min_size);

			cudaDeviceSynchronize();

			return device_version;


		}


		__device__ uint64_t get_full_hash(Key key){

			//todo seed
			return gallatin::hashers::MurmurHash64A(&key, sizeof(Key), 42);

		}

		__device__ node_type * get_new_node(){


			node_type * new_node = (node_type * ) gallatin::allocators::global_malloc(sizeof(node_type));

			if (new_node == nullptr) return nullptr;


			for (int i = 0; i < num_slots; i++){

				new_node->slots[i].key = defaultKey;

			}

			__threadfence();

			return new_node;

		}


		__device__ void attach_new_bucket(uint64_t index){

			node_type * new_node = get_new_node();

			directory.set_bucket_pointer(index, new_node);

		}


		__device__ bool insert(Key insert_key, Val insert_val){

			while (true){

				auto global_level = __ldcv(&directory.level);

				uint64_t bucket_hash = get_full_hash(insert_key);

				uint64_t clipped_hash = directory.mask_to_level(bucket_hash, global_level-1);
				node_type * my_bucket = directory.get_bucket_pointer(clipped_hash);
				uint16_t read_level;

				if (my_bucket->insert(insert_key, insert_val, global_level-1, read_level)){
					return true;
				}

				//otherwise maybe resize?

				if (read_level == global_level){

					if (directory.add_new_backing(global_level)){

						//resize was a success.
						//attempt to update s

					}

				} else {

					//split bucket!

					auto my_bucket_address = directory.get_bucket_address(directory.mask_to_level(bucket_hash, read_level-1));

				}


			}


		}



	};



}


}


#endif //end of resizing_hash guard