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


//including CG
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;


#define USE_ATOMICS 1
#define HT_PRINT 0


#define KEY_IS_HASH 0

//atomic version of the table
//ldcg is for chumps...
//this is to verify correctness before moving to looser instructions.
namespace gallatin {

namespace data_structs {


	using namespace gallatin::allocators;
	using namespace gallatin::utils;


		// //block type for extendible hashing

	template <typename Key, typename Val>
	struct extendible_key_val_pair {

		Key key;
		Val val;

	};



	//total size - 128 bytes
	//4 bytes for lock and size
	//num_keys bytes for fingerprints
	//16*num_keys bytes for key_vals.
	//7 or 14 are valid options
	//7 gives 128-17*7 = 5 bytes of leftover.
	//5 stash keys?


	template <typename Key, typename Val>
	struct bucket {

		//4 bytes
		uint fused_lock_size;  

		char tags[12];

		extendible_key_val_pair<Key, Val>[7];



	}

	//128*8?
	template <typename Key, typename Val>
	struct segment {

		using bucket_type = bucket<Key,Val>;

		bucket_type buckets[6];

		bucket_type stash[2];


		__device__ bool lock_triplet(uint main, uint secondary, uint tertiary){


			//three cases
			//in order normally


			if (tertiary < main && tertiary < secondary){
				//force unlock
				buckets[main].unlock();
				buckets[secondary].unlock();

				buckets[]
			}

		}

		__device__ bool insert(uint64_t hash, Key key, Val val){


			uint bucket = hash % 6;

			uint next_bucket = (hash + 1) % 6;

			bucket_type * my_primary = &buckets[bucket];

			bucket_type * my_secondary = &buckets[next_bucket];

			if (bucket > next_bucket){
				my_secondary.lock();
				my_primary.lock();
			} else {
				my_primary.lock();
				my_secondary.lock();
			}



			if (my_primary->upsert(key, val)) return true;
			if (my_secondary->upsert(key, val)) return true;

			if (my_primary->insert(key, val)) return true;

			if (my_secondary->insert(key, val)) return true;


			//main insertion failed, attempt displacement


		}


		__device__ void displace(bucket_type * primary, bucket_type * secondary, uint tertiary){

			my_tertiary = 

			int slot = secondary->get_displaceable_slot();

			if (slot != -1){
				//atempt move;
			}


		}

	}


	template<typename ht>
	__global__ void calculate_ext_ht_fill_kernel(ht * table, uint64_t * fill_count, uint64_t max_items){


		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;


		uint64_t fill = table->get_bucket_fill(tid);


		atomicAdd((unsigned long long int *)fill_count, (unsigned long long int) fill);


		if (table->get_bucket_present(tid)){

			atomicAdd((unsigned long long int *)&fill_count[1], 1ULL);

		}



	}


	template<typename ht>
	__global__ void free_ext_buckets(ht * table, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto bucket = table->get_bucket_for_fill(tid);

		if (bucket != nullptr){ global_free(bucket); }

	}

	template<typename ht>
	__global__ void free_ext_directory(ht * table, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto directory_ptr = table->directory[tid];

		if (directory_ptr != nullptr) { global_free(directory_ptr); }

	}

	// template <typename bucket_type>
	// bucket_iterator {

	// 	bucket_type * bucket;

	// 	uint64_t index;

	// 	__device__ bucket_iterator(bucket_type * ext_bucket){

	// 		index = 0;

	// 		bucket = ext_bucket;

	// 	}

	// }

	//insertion/query procedure

	//insert - query bucket size. If exact match, add item to bucket.
	//	otherwise, maybe need upsize. If the alt bucket is not visible, attempt an upsize.
	// if size > what I expect, no dice. Otherwise upsize occurs and items are moved.
	// Lock is used to control upsize, and exact atomicCAS on size is required to proceed.
	//queries - at worst need to check two buckets. Probably faster to always just check three.
	// check primary -> check secondary -> check primary.
	//	This dodges the case where keys are being shuttled. -> Secondary->primary is always correct:
	//	 keys must be entirely tranfered befor visbility in secondary is removed.
	template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, int num_pairs, int group_size>
	struct extendible_bucket {

		using my_type = extendible_bucket<Key, defaultKey, tombstoneKey, Val, num_pairs, group_size>;


		//determine sizing
		//16 bits always reserved for size.
		uint16_t size;

		uint16_t lock;

		static const uint64_t n_traversals = ((num_pairs-1)/group_size+1)*group_size;

		//do metadataBS?
		extendible_key_val_pair<Key, Val> slots [num_pairs];


		__device__ void init(uint16_t ext_size){

			size = ext_size;

			lock = 0;

			for (uint64_t i=0; i < num_pairs; i++){
				slots[i].key = defaultKey;
			}

			__threadfence();
		}

		__device__ Key resetPair(uint64_t index){

			return typed_atomic_exchange(&slots[index].key, tombstoneKey);


		}


		__device__ bool resetExact(uint64_t index, Key expected_key){

			return typed_atomic_write(&slots[index].key, expected_key, tombstoneKey);


		}


		__device__ bool insert_direct(int index, Key ext_key, Val ext_val){

			if (typed_atomic_write(&slots[index].key, defaultKey, ext_key)){
				typed_atomic_exchange(&slots[index].val, ext_val);
				return true;
			} else {

				#if HT_PRINT
				printf("Failed exchange!\n");
				#endif

				return false;
			}

		}



		__device__ int insert(Key ext_key, Val ext_val, cg::thread_block_tile<group_size> team){


			//first read size
			// internal_read_size = gallatin::utils::ldcv(&size);

			// //failure means resize has started...
			// if (internal_read_size != expected_size) return false;

			for (int i = team.thread_rank(); i < n_traversals; i+=team.size()){

				bool key_match = (i < num_pairs);

				Key loaded_key;

				if (key_match) loaded_key = gallatin::utils::ld_acq(&slots[i].key);

				//early drop if loaded_key is gone
				bool ballot = key_match && (loaded_key == defaultKey);

				auto ballot_result = team.ballot(ballot);

	       		while (ballot_result){

	       			ballot = false;

	       			const auto leader = __ffs(ballot_result)-1;

	       			if (leader == team.thread_rank()){


	       				ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
	       				if (ballot){
	       					typed_atomic_exchange(&slots[i].val, ext_val);
	       				}
	       			} 

	  

       				//if leader succeeds return
       				if (team.ballot(ballot)){
       					return __ffs(team.ballot(ballot))-1;
       				}
	       			

	       			//if we made it here no successes, decrement leader
	       			ballot_result  ^= 1UL << leader;

	       			//printf("Stalling in insert_into_bucket keys\n");

	       		}

	       		ballot = key_match && (loaded_key == tombstoneKey);

	       		ballot_result = team.ballot(ballot);

	       		while (ballot_result){

	       			ballot = false;

	       			const auto leader = __ffs(ballot_result)-1;

	       			if (leader == team.thread_rank()){
	       				ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);
	       				if (ballot){
	       					typed_atomic_exchange(&slots[i].val, ext_val);
	       				}
	       			} 

	  

       				//if leader succeeds return
       				if (team.ballot(ballot)){
       					return __ffs(team.ballot(ballot))-1;
       				}
	       			

	       			//if we made it here no successes, decrement leader
	       			ballot_result  ^= 1UL << leader;

	       			//printf("Stalling in insert_into_bucket\n");
	       			//printf("Stalling in insert_into_bucket tombstone\n");

	       		}


			}


			return -1;

		}

		__device__ Key peek_key(uint64_t index){

			return gallatin::utils::ldcv(&slots[index].key);

		}

		__device__ Val peek_val(uint64_t index){
			return gallatin::utils::ldcv(&slots[index].val);
		}


		__device__ uint16_t load_size_atomic(cg::thread_block_tile<group_size> team){

			return cg::invoke_one_broadcast(team, [&]() { return gallatin::utils::ldcv(&size); });

		}

		// __device__ bool query(Key ext_key, Val & ext_val, uint16_t expected_size, bool & other_check_needed){

		// 	//asserts that query may nnot be in another bucket.
		// 	uint16_t read_size = gallatin::utils::ldcv(&size);

		// 	if (read_size != expected_size){

		// 		other_check_needed = true;

		// 	}


		// 	for (int i = 0; i < num_pairs; i++){


		// 		if (slots[i].key == ext_key){

		// 			ext_val = gallatin::utils::ldcv(&slots[i].val);
		// 			return true;

		// 		}

		// 		//shortcut! Exit early as insert would have inserted here.
		// 		if (slots[i].key == defaultKey){
		// 			return false;
		// 		}

		// 	}

		// 	return false;

		// }

		__device__ bool query(Key ext_key, Val & ext_val, cg::thread_block_tile<group_size> team){

			//asserts that query may nnot be in another bucket.

			for (int i = team.thread_rank(); i < n_traversals; i+=team.size()){


				bool key_match = (i < num_pairs);

				Key loaded_key;

				if (key_match) loaded_key = gallatin::utils::ld_acq(&slots[i].key);

				bool ballot = (key_match && loaded_key == ext_key);


				auto ballot_result = team.ballot(ballot);
				if (ballot_result){
					//match!

					auto leader = __ffs(ballot_result)-1;

					if (team.thread_rank() == leader){
						ext_val = gallatin::utils::ld_acq(&slots[i].val);
					}

					ext_val = team.shfl(ext_val, leader);

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


		__device__ uint16_t stall_lock(){

			while (atomicCAS((unsigned short int *)&lock, (unsigned short int)0, (unsigned short int) 1) != 0){
				#if HT_PRINT
				printf("%llu Spinning on stall lock\n", gallatin::utils::get_tid());
				#endif
			}

		}


		__device__ uint16_t unlock(){
			atomicCAS((unsigned short int *)&lock, (unsigned short int)1, (unsigned short int) 0);
		}

		__device__ bool start_promotion(uint16_t promotion_size){

			stall_lock();

			if (promote_size(promotion_size-1) == promotion_size-1){
				return true;
			}


			unlock();

			return false;

		}


		__device__ void wait_on_bucket_promote(){


			while (atomicCAS((unsigned short int *)&lock, (unsigned short int)0, (unsigned short int) 0) != 0){
				#if HT_PRINT
				printf("Stalling\n");
				#endif
			}

		}


		__device__ int get_fill(){

			int count = 0;

			for (int i = 0; i < num_pairs; i++){

				if (slots[i].key != defaultKey && slots[i].key != tombstoneKey){
					count+=1;
				}

			}

			return count;



		}

	};



	template <typename T>
	__global__ void init_table_device(T * table){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid != 0) return;

		table->add_new_backing(0);

	}

	template <typename T>
	__global__ void set_table_buckets(T * table, uint64_t num_buckets, uint64_t min_bits){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= num_buckets) return;


		auto bucket = table->get_new_bucket(0);

		table->attach_bucket(bucket, tid);

	}


	template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, int items_per_bucket, uint64_t min_bits, uint64_t max_bits, int group_size>
	struct extendible_hash_table {

		using my_type = extendible_hash_table<Key, defaultKey, tombstoneKey, Val, items_per_bucket, min_bits, max_bits, group_size>;

		using bucket_type = extendible_bucket<Key, defaultKey, tombstoneKey, Val, items_per_bucket, group_size>;

		
		static const uint n_directory = max_bits-min_bits+1;
		//static const uint min_bits = gallatin::utils::numberOfBits(min_items-1)+1;

		//static const uint64_t nbits = 2*max_items; 

		static const uint64_t min_items = (1ULL << (min_bits));

		static const uint64_t max_items = (1ULL << (max_bits));

		//static const uint64_t max_hash_modulus = min_items << (n_directory-1);

		uint64_t level;
		uint64_t promote_level;

		//directory is 

		//upper level is an array of lower level: bucket_type *** (packed as array)
		//lowest level is an array of bucket *s bucket_type **
		bucket_type ** directory[max_bits-min_bits+1];


		static __host__ my_type * generate_on_device(){


			printf("Min bits: %lu, Max bits: %lu, n_directory: %lu, min_items: %lu, max_items: %lu, CG: %d\n", min_bits, max_bits, n_directory, min_items, max_items, group_size);

			printf("Size of bucket: %llu, Max size: %fGB\n", sizeof(bucket_type), 1.0*(max_items*(sizeof(bucket_type)+sizeof(bucket_type*)))/(1024ULL*1024*1024));

			my_type * host_version = gallatin::utils::get_host_version<my_type>();



			//host_version[0] = default_host_version;

			//host_version->live_bits = gallatin::utils::get_device_version<uint64_t>(host_version->nbits);

			host_version->promote_level = 0;
			host_version->level = 0;


			//printf("Live bits %llu, max items: %llu\n", host_version->nbits, max_items);

			//cudaMemset(host_version->live_bits, 0ULL, sizeof(uint64_t)*max_items*2);


			my_type * device_version = gallatin::utils::move_to_device(host_version);

			init_table_device<my_type><<<1,1>>>(device_version);

			set_table_buckets<my_type><<<(min_items-1)/256+1,256>>>(device_version, min_items, min_bits);

			cudaDeviceSynchronize();

			return device_version;


		}


		static __host__ void free_on_device(my_type * dev_version){

			free_ext_buckets<my_type><<<(max_items-1)/256+1, 256>>>(dev_version, max_items);
			free_ext_directory<my_type><<<(n_directory-1)/256+1, 256>>>(dev_version, n_directory);

			cudaDeviceSynchronize();

			cudaFree(dev_version);

			cudaDeviceSynchronize();



		}

		//v2 - to prevent stalling, all threads must go through Gallatin?
		//attempt load.
		//if below what we need, upgrade.
		//promoting from prev_size->prev_size+1;
		__device__ bool add_new_backing(uint64_t prev_size){

			//make this atomic check?

			//uint64_t local_level = atomicAdd((unsigned long long int *)&level, 0ULL);
			uint64_t local_level = gallatin::utils::ld_acq(&level);

			while (local_level < prev_size+1){

				if (atomicCAS((unsigned long long int *)&promote_level, (unsigned long long int) prev_size, (unsigned long long int) prev_size+1) == prev_size){

					uint64_t new_size = 1ULL << (min_bits+prev_size-1);
					if (prev_size <= 1){
						new_size = (1ULL << (min_bits));
					}

					//entered lock?
					#if HT_PRINT
					printf("Asking for allocation of size %llu\n", sizeof(bucket_type *)*new_size);
					#endif
					bucket_type ** new_backing = (bucket_type **) global_malloc((sizeof(bucket_type *)*new_size));

					if (new_backing == nullptr){
						#if HT_PRINT
						printf("Err: failed to allocate %llu pointers\n", sizeof(bucket_type *)*new_size);
						#endif
						asm volatile("trap;");
					} else {
						#if HT_PRINT
						printf("Acquired malloc %llx\n", new_backing);
						#endif
					}


					if (!atomicCAS((unsigned long long int *)&directory[prev_size], 0ULL, (unsigned long long int)new_backing) == 0ULL){
						#if HT_PRINT
						printf("Weird behavior\n");
						#endif
						asm volatile("trap;");
					}

					while (atomicCAS((unsigned long long int *)&level, (unsigned long long int) prev_size, (unsigned long long int) prev_size+1) != prev_size);
					
					//local_level = atomicAdd((unsigned long long int *)&level, 0ULL);
					local_level = gallatin::utils::ld_acq(&level);


				} else {
					//local_level = atomicAdd((unsigned long long int *)&level, 0ULL);
					local_level = gallatin::utils::ld_acq(&level);
				}

			}

			return;



			// 	printf("%llu Spinning, local level %llu, need %llu\n", gallatin::utils::get_tid(), local_level, prev_size+1);

			// 	uint64_t new_size = 1ULL << (min_bits+prev_size-1);
			// 	if (prev_size <= 1){
			// 		new_size = (1ULL << (min_bits));
			// 	}

			// 	printf("Asking for allocation of size %llu\n", sizeof(bucket_type *)*new_size);
			// 	bucket_type ** new_backing = (bucket_type **) global_malloc((sizeof(bucket_type *)*new_size));

			// 	if (new_backing == nullptr){
			// 		printf("Err: failed to allocate %llu pointers\n", new_size);
			// 		continue;
			// 	} else {
			// 		printf("Acquired malloc %llx\n", new_backing);
			// 	}

			// 	if (atomicCAS((unsigned long long int *)&directory[prev_size], 0ULL, (unsigned long long int)new_backing) == 0ULL){

			// 		//should take only one.
			// 		while (atomicCAS((unsigned long long int *)&level, (unsigned long long int) prev_size, (unsigned long long int) prev_size+1) < prev_size){
			// 			printf("Spinning on transition: %llu->%llu\n", prev_size, prev_size+1);
			// 		}

			// 		//force update to be read
			// 		gallatin::utils::st_rel(&level, prev_size+1);
			// 		__threadfence();

			// 		printf("Store occurred for size %lu\n", prev_size+1);
			// 		return;

			// 	} else {

			// 		while (atomicCAS((unsigned long long int *)&level, (unsigned long long int) prev_size, (unsigned long long int) prev_size+1) < prev_size){
			// 			printf("Spinning on transition: %llu->%llu\n", prev_size, prev_size+1);
			// 		}

			// 		__threadfence();
			// 		global_free(new_backing);
			// 	}


			// }

		}


		__device__ uint64_t generate_clipped_hash(Key key){

				#if KEY_IS_HASH

				return clip_hash_to_max_size(key);

				#else

				return clip_hash_to_max_size(get_full_hash(key));

				#endif

		}


		__device__ uint64_t cooperative_get_hash(Key key, cg::thread_block_tile<group_size> & team){

			return cg::invoke_one_broadcast(team, [&] () { return generate_clipped_hash(key); });

		}


		__device__ uint64_t cooperative_get_global_level(cg::thread_block_tile<group_size> & team){

			return cg::invoke_one_broadcast(team, [&] () { return gallatin::utils::ld_acq(&level)-1; });
								

		}

		// 	while (atomicCAS((unsigned long long int *)&promote_level, 0ULL, 1ULL) != 0ULL){

		// 		printf("Stalling on lock: %llu, %llu\n", promote_level, gallatin::utils::ld_acq(&level));
		// 	}


		// 	uint64_t local_level = gallatin::utils::ld_acq(&level);

		// 		if (local_level > prev_size){
		// 			atomicCAS((unsigned long long int *)&promote_level, 1ULL, 0ULL);
		// 			return false;
		// 		}

		// 		printf("Entered promote level lock %lu->%lu\n", prev_size, prev_size+1);

		// 		//progression = 
		// 		//256
		// 		//256
		// 		//512
		// 		uint64_t new_size = 1ULL << (min_bits+prev_size-1);
		// 		if (prev_size <= 1){
		// 			new_size = (1ULL << (min_bits));
		// 		}
				

		// 		printf("Starting upgrade on %llu\n", new_size);


		// 		bucket_type ** new_backing = (bucket_type **) global_malloc((sizeof(bucket_type *)*new_size));

		// 		if (new_backing == nullptr){
		// 			printf("Failed to get new backing for size %llu\n", new_size);
		// 		} else {
		// 			printf("Acquired backing for size %llu: %llx\n", new_size, (uint64_t) new_backing);
		// 		}

		// 		directory[prev_size] = new_backing;

		// 		__threadfence();

		// 		//force updates to be visible in order.
		// 		while (atomicCAS((unsigned long long int *)&level, (unsigned long long int) prev_size, (unsigned long long int) prev_size+1) != prev_size){

		// 			printf("Stalling on add backing from %lu -> %lu\n", prev_size, prev_size+1);

		// 		}

		// 		printf("upgrade done\n");
		// 		atomicCAS((unsigned long long int *)&promote_level, 1ULL, 0ULL);
		// 		return true;

		// }

		__device__ bucket_type * get_new_bucket(uint16_t size){


			bucket_type * new_bucket = (bucket_type *) global_malloc(sizeof(bucket_type));

			if (new_bucket == nullptr){
				#if HT_PRINT
				printf("Failed to allocate bucket\n");
				#endif
				//new_bucket = (bucket_type *) global_malloc(sizeof(bucket_type));
				asm volatile("trap;");

			}

			new_bucket->init(size);

			return new_bucket;

		}


		__device__ bool attach_bucket(bucket_type * bucket, uint64_t position){


			uint64_t directory_index = get_directory_index(position);

			uint64_t local_position = get_local_position(position,directory_index);

			if (directory[directory_index] == nullptr){
				//printf("Bad directory\n");
				return false;
			}

			uint64_t result = atomicCAS((unsigned long long int *)&directory[directory_index][local_position], 0ULL, (unsigned long long int)bucket);

			return (result == 0ULL);

		}


		//improve this later
		//assumes clipped hash in range of table.
		__device__ uint64_t get_directory_index(uint64_t clipped_hash){

			uint64_t index = 0;

			uint64_t items_covered = min_items;

			while (true){

				if (clipped_hash < items_covered){

					return index;

				}

				index+=1;

				if (index == 1) items_covered = min_items;
				items_covered = items_covered << 1;

			}



		}


		__device__ uint64_t get_local_position(uint64_t clipped_hash, uint64_t index){

			if (index == 0) return clipped_hash;

			if (index == 1) return clipped_hash - min_items;

			uint64_t items_at_level_below = min_items + min_items << (index-2);

			return clipped_hash - items_at_level_below;


		}



		__device__ uint64_t clip_hash_to_max_size(uint64_t hash){


			return hash % (max_items << 1);

		}

		__device__ uint64_t clip_to_global_level(uint64_t level, uint64_t clipped_hash){


			return clipped_hash & BITMASK((level+min_bits));

		}

		__device__ uint64_t get_full_hash(Key key){

			//todo seed
			return gallatin::hashers::MurmurHash64A(&key, sizeof(Key), 42);
		}


		__device__ bucket_type * get_bucket_from_index(uint64_t index, bool load_atomic=false){

			uint64_t directory_index = get_directory_index(index);

			uint64_t local_position = get_local_position(index,directory_index);

					//check if directory exists - iteratively refine untile smaller key reached


			bucket_type ** global_read_directory = directory[directory_index];

			//to get to this point, the directory must be set somewhere.
			//loop until the cache mechanism detects and corrects
			while (global_read_directory == nullptr){

				//printf("Stalling in read of global directory\n");

				//BLEGH
				//global_read_directory = (bucket_type ** ) atomicCAS((unsigned long long int *)&directory[directory_index],0ULL, 0ULL);

				//better if this works.
				global_read_directory = (bucket_type **) gallatin::utils::ld_acq((uint64_t *)&directory[directory_index]);


				#if HT_PRINT
				printf("Looping %lu\n", gallatin::utils::get_tid());
				#endif

				//extra safety check - if less than global level, drop.

			}

			return (bucket_type *) gallatin::utils::ld_acq((uint64_t *)&global_read_directory[local_position]);


			// if (load_atomic && global_read_directory[local_position] == nullptr){

			// 	return (bucket_type * ) atomicCAS((unsigned long long int *)&global_read_directory[local_position], 0ULL, 0ULL);

			// } else {
			// 	return global_read_directory[local_position];
			// }

		}


		__device__ bucket_type * get_bucket_for_fill(uint64_t index, bool load_atomic=false){

			uint64_t directory_index = get_directory_index(index);

			uint64_t local_position = get_local_position(index,directory_index);

					//check if directory exists - iteratively refine untile smaller key reached


			bucket_type ** global_read_directory = directory[directory_index];

			//to get to this point, the directory must be set somewhere.
			//loop until the cache mechanism detects and corrects
			if (global_read_directory == nullptr){

				//printf("Stalling in read of global directory\n");

				return nullptr;

			}



			if (load_atomic && global_read_directory[local_position] == nullptr){

				return (bucket_type * ) atomicCAS((unsigned long long int *)&global_read_directory[local_position], 0ULL, 0ULL);

			} else {
				return global_read_directory[local_position];
			}

		}

		__device__ bool insert(Key key, Val val, cg::thread_block_tile<group_size> team){


			uint64_t hash = cooperative_get_hash(key, team);
			uint64_t global_level = cooperative_get_global_level(team);
			uint64_t local_level = global_level;

			bucket_type * primary_bucket;
			

			while (true){

				#if HT_PRINT
				printf("%llu Looping in main\n", gallatin::utils::get_tid());
				#endif

				
				//broadcast info

				
				#if HT_PRINT
				if (local_level >= n_directory){
					printf("Weird local_level: %lu\n", local_level);
				}

				if (local_level > global_level){
					printf("Weird local_level > global_level\n");
				}
				#endif
				//should this be -1?

				//local level is too large sometimes here. How?

				//force refinement if not valid
				uint64_t bucket_index = clip_to_global_level(local_level, hash);

				if (team.thread_rank() == 0){
					primary_bucket = get_bucket_from_index(bucket_index, true);
				}
				primary_bucket = team.shfl(primary_bucket, 0);

				//base case - bucket should always be 0.
				// if (local_level == 0){

				// 	primary_bucket = get_bucket_from_index(bucket_index, true);

				// 	if (primary_bucket == nullptr){
				// 		printf("Bug setting primary bucket in index %llu\n", bucket_index);
				// 		return false;
				// 	}

				// }

				//printf("Tid %llu Looping on local level %lu\n", gallatin::utils::get_tid(), local_level);

				if (primary_bucket != nullptr){

					int insert_slot = primary_bucket->insert(key, val, team);
					if (insert_slot != -1){

						//check size for rollback


						auto bucket_size = primary_bucket->load_size_atomic(team);

						if (clip_to_global_level(bucket_size, hash) == bucket_index){
							return true;

						} else {

							//rollback.

							if (cg::invoke_one_broadcast(team, [&] () { return primary_bucket->resetExact(insert_slot, key); })){

								__threadfence();
								local_level = local_level+1;


								global_level = cg::invoke_one_broadcast(team, [&] () { return gallatin::utils::ld_acq(&level)-1; });
								//global_level = atomicAdd((unsigned long long int *)&level, 0ULL)-1;

								#if HT_PRINT
								if (local_level > global_level){
									printf("Upper track generates bug\n");
								}
								#endif



								continue;

							}

							return true;

						}





						return true;
					} else {

						//fail! see if we can promote the bucket.


						//things needed
						//1. bucket size.
						//2. global size.
						//3. local level.


						//procedure
						//if local level and global bucket size are n_directory return.
						//global read bucket size
						//if new bucket should be added
						//check if should add new backing
						//if true expand


						//after keys are moved up local level if not cap.
						//reload


						auto primary_bucket_size = primary_bucket->load_size_atomic(team);

						if (local_level == (n_directory-1) && primary_bucket_size == (n_directory-1)){

							//table is full!
							return false;
						}

						//early drop - primary bucket can't resize, but maybe another bucket can
						//this shouldn't happen but who knows? We'll leave it in for now.
						if (primary_bucket_size == (n_directory-1)){

							#if HT_PRINT
							printf("Triggering local resize without bucket upsize\n");
							#endif
							local_level = local_level+1;

							//global_level = atomicAdd((unsigned long long int *)&level, 0ULL)-1;
							global_level = cooperative_get_global_level(team);
							#if HT_PRINT
							if (local_level > global_level){
								printf("Mid track generates bug\n");
							}
							#endif

							__threadfence();
							continue;
						}

						//proceeding with resize Load external data needed
						global_level = cooperative_get_global_level(team);


						if (primary_bucket_size == global_level){
							//implies global_level < n_directory;
							//therefore you can (and should) upsize safely.
							cg::invoke_one(team, [&] () { add_new_backing(primary_bucket_size+1); });

						}

						//at this point, we should upsize the bucket.
						maybe_add_new_bucket(bucket_index, primary_bucket_size+1, primary_bucket, team);

						//at this point, reload variables


						

						__threadfence();
						global_level = cooperative_get_global_level(team);
						//global_level = atomicAdd((unsigned long long int *)&level, 0ULL)-1;


						if (local_level != (n_directory-1) && local_level < global_level) local_level+=1;

						#if HT_PRINT
						if (local_level > global_level){
							printf("Final track generates bug: %lu > %lu\n",local_level, global_level);
						}
						#endif

						continue;


						//start of extension

						// local_level = local_level+1;

						// __threadfence();
						// //reload global level...
						// global_level = gallatin::utils::ld_acq(&level)-1;

						// if (local_level > global_level){

						// 	//can't expand.
						// 	if (local_level >= n_directory) return false;


						// 	add_new_backing(global_level+1);

						// 	__threadfence();

						// 	global_level = gallatin::utils::ld_acq(&level)-1;

						// 	//after new backing, fall back to re-read global.
						// 	continue;
						// }

						//regenerate index
						//uint64_t alt_index = clip_to_global_level(local_level, hash);

						//if (alt_index == local_level) break;

						//auto alt_bucket = get_bucket_from_index(alt_index);


						//printf("Attempting move\n");
						//promote size.

						// if (local_level >= n_directory) return false;

						// auto primary_bucket_size = primary_bucket->load_size_atomic();

						// if (primary_bucket_size >= global_level){


						// 	if (primary_bucket_size+1 >= n_directory) return false;

						// 	//printf("Adding Backing for b")

						// 	add_new_backing(primary_bucket_size+1);
							
						// 	continue;

						// } 

	


					}

				} else {
					//refine
					local_level = local_level-1;
				}

				



			

			}


			return false;

		}


		//conditions
		//bucket start
		// prep size is size
		// true size is size-1
		//bucket end
		// prep size is size
		// true size is size-1 
		__device__ void move_into_new_bucket(uint64_t start_index, uint64_t alt_index, uint64_t promotion_size, bucket_type * start, bucket_type * end, cg::thread_block_tile<group_size> team){

			uint64_t moved_keys = 0;
			//iterate through bucket.

			uint64_t items_moved = 0ULL;

			for (int i = 0; i < items_per_bucket; i++){

				//global read of key - non-destructive.
				auto currentKey = cg::invoke_one_broadcast(team, [&](){ return start->peek_key(i); });

				//done! all later keys MUST be null;
				if (currentKey == defaultKey) return;

				if (currentKey != tombstoneKey){

					uint64_t hash = cooperative_get_hash(currentKey, team);

					//uint64_t hash = clip_hash_to_max_size(get_full_hash(currentKey));

					uint64_t index = clip_to_global_level(promotion_size, hash);

					if (index == start_index) continue;

					#if HT_PRINT
					if (index != alt_index){
						printf("Expanded index is not correct on insertion, %lu != %lu or %lu\n", index, start_index, alt_index);
					}
					#endif


					items_moved |= SET_BIT_MASK(i);
					moved_keys+=1;

					auto currentVal = cg::invoke_one_broadcast(team, [&] (){ return start->peek_val(i); });

					//at this point, valid.
					//start transfer
					end->insert(currentKey, currentVal, team);


				}

			}
			#if HT_PRINT
			printf("Moved %lu keys\n", moved_keys);
			#endif

			for (int i = 0; i < items_per_bucket; i++){

				if ((SET_BIT_MASK(i) & items_moved) == 0) continue;

				auto currentKey = cg::invoke_one_broadcast(team, [&] () { return start->peek_key(i); });

				if (currentKey != tombstoneKey && currentKey != defaultKey){

										
					uint64_t hash = cooperative_get_hash(currentKey,team);

					uint64_t index = clip_to_global_level(promotion_size, hash);

					if (index == alt_index){

						cg::invoke_one(team, [&] () { start->resetPair(i); });

					}


				}


			}

		}

		//cooperative version that moves group_size items at a time.
		__device__ void move_into_new_bucket_coop(uint64_t start_index, uint64_t alt_index, uint64_t promotion_size, bucket_type * start, bucket_type * end, cg::thread_block_tile<group_size> team){

			uint64_t moved_keys = 0ULL;
			//iterate through bucket.

			uint64_t items_moved = 0ULL;

			for (int i = team.thread_rank(); i < start->n_traversals; i+=team.size()){


				Key currentKey = tombstoneKey;


				bool key_match = (i < items_per_bucket);


				if (key_match) currentKey = start->peek_key(i);

				//if (currentKey == defaultKey) continue;

				uint64_t hash = generate_clipped_hash(currentKey);

				uint64_t index = clip_to_global_level(promotion_size, hash);

				bool moving = key_match && (currentKey != tombstoneKey) && (currentKey != defaultKey) && (index != start_index);

				auto ballot_moving = team.ballot(moving);

				auto keys_below = __popc(ballot_moving & BITMASK(team.thread_rank()));



				if (moving){

					items_moved |= SET_BIT_MASK(i);

					auto currentVal = start->peek_val(i);

					if (!end->insert_direct(moved_keys+keys_below, currentKey, currentVal)){
						printf("Bucket %llx Failed to insert from %d to %llu\n", (unsigned long long int) start, i, moved_keys+keys_below);
					}

				}

				//update tracking.
				moved_keys += __popc(ballot_moving);


			}


			team.sync();


			bool bucket_attached = cg::invoke_one_broadcast(team, [&] () { return attach_bucket(end, alt_index); } );

			if (!bucket_attached){

				printf("Failed to attach!\n");

				asm volatile("trap;");
			}

			//unset round
			for (int i = team.thread_rank(); i < start->n_traversals; i+=team.size()){		

				if ((SET_BIT_MASK(i) & items_moved) == 0) continue;

				start->resetPair(i);

			}



		}


		__device__ void maybe_add_new_bucket(uint64_t index,  uint64_t promotion_size, bucket_type * primary_bucket, cg::thread_block_tile<group_size> team){


			if (promotion_size >= n_directory){
				//printf("Promotion size %lu >= %lu\n", promotion_size, n_directory);
				return;
			}

			if (cg::invoke_one_broadcast(team, [&] () { return primary_bucket->start_promotion(promotion_size);})){

				//printf("Promotion started: resizing from %lu -> %lu\n", promotion_size-1, promotion_size);


				//determine updgrade position...
				//current size is promotion size -1.

				uint64_t current_n_elements = (1ULL << (min_bits+promotion_size-1));


				uint64_t alt_index = index+current_n_elements;

				#if HT_PRINT
				if ((current_n_elements & index) != 0){
					printf("Weird index %lu, size %lu", index, current_n_elements);
				}
				#endif


				auto alt_bucket = cg::invoke_one_broadcast(team, [&] () { return get_new_bucket(promotion_size);});


				//bool failed_attach = false;


				// if (team.thread_rank() == 0){

				// 	alt_bucket->stall_lock();


				// 		//printf("Bucket added\n");

				// }

				cg::invoke_one(team, [&] () { alt_bucket->stall_lock(); });

				// if (team.ballot(failed_attach)) {

				// 	#if HT_PRINT
				// 	printf("New bucket set failed\n");
				// 	#endif

				// 	asm volatile("trap;");


				// }

				//bucket attached, begin promotion process
				move_into_new_bucket_coop(index, alt_index, promotion_size, primary_bucket, alt_bucket, team);


				if (team.thread_rank() == 0){
					primary_bucket->unlock();

					alt_bucket->unlock();
				}

				team.sync();


				//printf("Bucket move finished\n");

				__threadfence();



			} else {

				cg::invoke_one(team, [&] () { primary_bucket->wait_on_bucket_promote(); });

				team.sync();

			}






		}

		//step through levels, looking for key
		//we must probe at most 2 buckets - this should always be verifiable by stepping down through the buckets
		//do we need to step up occassionally?
		__device__ bool query(Key key, Val & val, cg::thread_block_tile<group_size> team){


			uint64_t hash = cooperative_get_hash(key, team);
			uint64_t global_level = cooperative_get_global_level(team);
			uint64_t local_level = global_level;



			while (true){
				

				//should this be -1?

				//local level is too large sometimes here. How?

				//force refinement if not valid
				uint64_t bucket_index = clip_to_global_level(local_level, hash);

				auto primary_bucket = cg::invoke_one_broadcast(team, [&] () { return get_bucket_from_index(bucket_index, true); });


				//printf("Tid %llu Looping on local level %lu\n", gallatin::utils::get_tid(), local_level);

				if (primary_bucket != nullptr){

					if (primary_bucket->query(key, val, team)){
						return true;
					} else {

						if (local_level != 0){

							uint64_t alt_bucket_index = clip_to_global_level(local_level-1, hash);

							auto secondary_bucket = cg::invoke_one_broadcast(team, [&] () { return get_bucket_from_index(alt_bucket_index, true); }); 

							#if HT_PRINT
							if (secondary_bucket == nullptr){
								printf("BUG in ht query.\n");
							}
							#endif



							if (!secondary_bucket->query(key, val, team)){

								return false;

							}

							return true;



						} else {
							return false;
						}


					}

				} else {
					//refine
					local_level = local_level-1;
				}

				



			

			}


			return false;



		}

		__host__ double calculate_fill(bool max_size=true){

			uint64_t * items_in_table;

			cudaMallocManaged((void **)&items_in_table, sizeof(uint64_t)*2);

			cudaDeviceSynchronize();

			items_in_table[0] = 0;
			items_in_table[1] = 0;

			cudaDeviceSynchronize();


			calculate_ext_ht_fill_kernel<my_type><<<(max_items-1)/256+1,256>>>(this, items_in_table, max_items);

			cudaDeviceSynchronize();


			double total_fill;

			if (max_size){
				total_fill = (1.0*items_in_table[0])/(max_items*items_per_bucket);
			} else {
				total_fill = (1.0*items_in_table[0])/(items_in_table[1]*items_per_bucket);
			}

			cudaFree(items_in_table);

			return total_fill;


		}

		__device__ int get_bucket_fill(uint64_t bucket_id){

			auto bucket = get_bucket_for_fill(bucket_id);

			if (bucket == nullptr) return 0;


			return bucket->get_fill();


		}

		__device__ bool get_bucket_present(uint64_t bucket_id){

			auto bucket = get_bucket_for_fill(bucket_id);

			return bucket != nullptr;

		}

		//helper for pulling tiles
		__device__ __inline__ cg::thread_block_tile<group_size> get_my_tile(){

		auto thread_block = cg::this_thread_block();

  	 	cg::thread_block_tile<group_size> my_tile = cg::tiled_partition<group_size>(thread_block);

  	 	return my_tile;
 
		}

	};



}


}


#endif //end of resizing_hash guard