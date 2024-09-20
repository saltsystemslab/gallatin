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

//updated ht that uses a bitarray merged with size
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



	template <typename T, int lower_bits>
	struct packed_pointer {

		using my_type = packed_pointer<T, lower_bits>;

		uint64_t bits;


		__device__ static my_type pack_together(T * pointer, uint size){

			uint64_t pointer_as_bits = (uint64_t) pointer;

			return my_type(pointer_as_bits | size);

		}

		__device__ packed_pointer(){
			bits = 0ULL;
		}

		__device__ packed_pointer(uint64_t ext_bits){

			bits = ext_bits;

		}

		//clip lower bits;
		//as long as pointers are aligned 2^lower_bits this is safe.
		__device__ T * get_pointer(){

			return (T *) (bits & (~BITMASK(lower_bits)));

		}

		//return size by taking only lower bits.
		__device__ uint get_size(){

			return bits & BITMASK(lower_bits);

		}

		__device__ explicit operator unsigned long long int() const { return bits; }


		__device__ bool operator==(const T * rhs){

			return rhs == get_pointer();

		}

		__device__ bool operator!=(const T * rhs){

			return rhs != get_pointer();

		}

		__device__ T * operator->(){
			return get_pointer();
		}


	};


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

	//set yourself to nullptr if you do not detect that the bucket is the same level as you
	//This occurs IFF there is difference in your level + bucket level where bucket level is smaller than your level
	//To detect this, call load and clip to global level.
	template<typename ht>
	__global__ void clip_ext_buckets(ht * table, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto bucket = table->get_bucket_for_fill(tid);

		auto bucket_level = bucket->load_size_atomic_singleton();

		if (table->clip_to_global_level(bucket_level, tid) != tid){

			table->swap_bucket_in_index(tid, bucket, nullptr);

		}

	}


	template<typename ht>
	__global__ void free_ext_buckets(ht * table, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto bucket = table->get_bucket_for_fill(tid);

		//need to check for double assignment...

		if (bucket != nullptr){ global_free(bucket); }

	}

	template<typename ht>
	__global__ void free_ext_directory(ht * table, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto directory_ptr = table->directory[tid];

		if (directory_ptr != nullptr) { global_free(directory_ptr); }

	}


	template <typename ht>
	__global__ void set_table_pointers(ht * table, uint64_t min_items, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();


		if (tid < min_items) return;

		if (tid >= max_items) return;


		auto lower_bucket_tid = table->clip_to_global_level(0, tid);

		//this doesn't trigger.
		#if HT_PRINT
		if (lower_bucket_tid != (tid % min_items)){
			printf("This assumption was wrong\n");
		}
		#endif

		auto my_lower_bucket = table->get_bucket_from_index(lower_bucket_tid);

		//this doesn't trigger.
		if (my_lower_bucket.get_size() != 0){
			printf("Size is off\n");
		}

		auto lower_bucket_size = my_lower_bucket.get_pointer()->load_size_atomic_singleton();

		if (lower_bucket_size != 0){
			printf("Internal bucket size is wrong\n");
		}

		//auto size = my_lower_bucket->load_size_atomic_singleton();

		//auto packed_lower = table->pack_together(my_lower_bucket, size);

		auto packed_nullptr = table->pack_together(nullptr, 0U);

		table->swap_bucket_in_index(tid, packed_nullptr, my_lower_bucket);



	}


	//verify that each pointer points to an actual table entry.
	template <typename ht>
	__global__ void verify_starting_conditions(ht * table, uint64_t max_items){


		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= max_items) return;

		auto upper_bucket_wrapped = table->get_bucket_from_index(tid);

		auto size = upper_bucket_wrapped.get_size();

		auto ptr = upper_bucket_wrapped.get_pointer();

		uint64_t lower_tid = table->clip_to_global_level(size, tid);

		auto lower_bucket_wrapped = table->get_bucket_from_index(tid);

		auto lower_size = lower_bucket_wrapped.get_size();

		auto lower_pointer = lower_bucket_wrapped.get_pointer();

		auto bucket_size = lower_pointer->load_size_atomic_singleton();

		if (lower_pointer != ptr){
			printf("CHECK_FAIL: pointer mismatch %llu != %llu\n", ptr, lower_pointer);
		}

		if (size != lower_size || lower_size != bucket_size){
			printf("CHECK_FAIL: size mismatch %u != %u != %u\n", size, lower_size, bucket_size);
		}

		//at the end of this, every pair of linked addresses are in agreement - they point to the lowest level bucket with size 0.

	}



	template <typename ht>
	__global__ void prealloc_all_buckets(ht * table, uint64_t min_items, uint64_t max_items){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid < min_items) return;
		if (tid >= max_items) return;

		uint64_t directory_level = table->get_directory_index(tid);

		auto new_bucket = table->get_new_bucket(directory_level);

		table->prealloc_directory[tid] = new_bucket;

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
		// uint16_t size;

		// uint16_t lock;
		//64bits
		//1 bit lock
		//7 bits size
		//56 bits for 7 8-bit tags
		//these allow for rapid check of slot occupation
		//resulting in faster allocations.
		uint64_t packed_size_lock_tags;


		static const uint64_t n_traversals = ((num_pairs-1)/group_size+1)*group_size;

		extendible_key_val_pair<Key, Val> slots [num_pairs];

		static_assert(num_pairs <= 7);


		__device__ void init(uint8_t ext_size){

			packed_size_lock_tags = ((uint64_t) (ext_size)) << 56;

			//lock = 0;

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


		__device__ uint64_t generate_tag_mask(uint8_t key_tag){


			return 0x0001010101010101ULL * key_tag;


		}


		__device__ uint8_t generate_tag(Key ext_key){


			return (ext_key & BITMASK(8));

		}

		//takes in packed uint8_t and unsets tags.
		__device__ void wipe_tags(uint8_t wiped_tags){

			//expands bits to 8 every 8.
			uint64_t large_tag = ((uint64_t) ~wiped_tags)*8;

			//sets 0xff in each saved bitmask
			uint64_t set_bitmask = large_tag*(255U);

			atomicAnd((unsigned long long int *)&packed_size_lock_tags, set_bitmask);


		}


		__device__ bool write_tag(int index, uint8_t tag, uint64_t existing_tags){

			uint8_t * tags_as_uint = (uint8_t *) existing_tags;

			uint64_t packed_tag = ((uint64_t) tag) << (index*8);

			while (tags_as_uint[index] == 0){

				existing_tags = atomicCAS((unsigned long long int *)&packed_size_lock_tags, (unsigned long long int)existing_tags, (unsigned long long int) (existing_tags | packed_tag));

			}

			return tags_as_uint[index] == tag;


		}


		__device__ int insert(Key ext_key, Val ext_val, cg::thread_block_tile<group_size> team){


			//first read size
			// internal_read_size = gallatin::utils::ldcv(&size);

			// //failure means resize has started...
			// if (internal_read_size != expected_size) return false;

			uint64_t size_and_tags = load_size_atomic(team);


			//generate_tag_mask

			uint8_t target_tag = generate_tag(ext_key);

			//uint64_t tag_mask = generate_tag_mask(target_tag);

			//need to find exact match.
			//xor leaves 0's on exact match.
			//uint64_t match = (size_and_tags ^ tag_mask);

			uint8_t * size_and_tags_as_uint = (uint8_t *) &size_and_tags;

			//uint8_t * match_as_uint = (uint8_t *) &match;

			for (int i = team.thread_rank(); i < n_traversals; i+=team.size()){


				bool key_match = (i < num_pairs);

				bool exist_match = (size_and_tags_as_uint[i] == target_tag);

				bool empty_match = (size_and_tags_as_uint[i] == 0U);

				bool loaded = (key_match && (exist_match || empty_match));


				auto ballot_result = team.ballot(loaded);

				while (ballot_result){

	       			ballot = false;

	       			const auto leader = __ffs(ballot_result)-1;

	       			if (leader == team.thread_rank()){



	       				if (exist_match){
	       					//update val with my val
	       					if (gallatin::utils::ld_acq(&slots[i].key) == ext_key){
	       						gallatin::utils::st_rel(&slots[i].val, ext_val);
	       						ballot = true;
	       					} else {
	       						ballot = false;
	       					}

	       				} else {

							//writing tag locks you into position.
							//other threads need to wait.
	       					ballot = write_tag(i, target_tag);

		       				if (ballot){
		       					gallatin::utils::st_rel(&slots[i].key, ext_key);
		       					gallatin::utils::st_rel(&slots[i].val, ext_val);
		       				}

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

			}


			return -1;

		}

		__device__ Key peek_key(uint64_t index){

			return gallatin::utils::ldcv(&slots[index].key);

		}

		__device__ Val peek_val(uint64_t index){
			return gallatin::utils::ldcv(&slots[index].val);
		}


		__device__ uint64_t load_size_atomic(cg::thread_block_tile<group_size> team){

			return cg::invoke_one_broadcast(team, [&]() { return gallatin::utils::ld_acq(&packed_size_lock_tags); });

		}

		__device__ uint64_t load_size_atomic_singleton(){

			return gallatin::utils::ld_acq(&packed_size_lock_tags);

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


		for (int i = 0; i < table->n_directory; i++){

			table->add_new_backing(i);

		}

		//table->add_new_backing(0);



	}

	template <typename T>
	__global__ void set_table_buckets(T * table, uint64_t num_buckets){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= num_buckets) return;


		auto bucket = table->get_new_bucket(0);

		auto packed = table->pack_together(bucket, 0);

		table->attach_bucket(packed, tid);

	}


	template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, int items_per_bucket, uint64_t min_bits, uint64_t max_bits, int group_size>
	struct extendible_hash_table {

		using my_type = extendible_hash_table<Key, defaultKey, tombstoneKey, Val, items_per_bucket, min_bits, max_bits, group_size>;

		using bucket_type = extendible_bucket<Key, defaultKey, tombstoneKey, Val, items_per_bucket, group_size>;

		//bucket pointer type
		using bpt = packed_pointer<bucket_type, 5>;
		
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
		bpt directory[max_items];

		bucket_type * prealloc_directory[max_items];


		static __host__ my_type * generate_on_device(){


			printf("Min bits: %lu, Max bits: %lu, n_directory: %lu, min_items: %lu, max_items: %lu, CG: %d\n", min_bits, max_bits, n_directory, min_items, max_items, group_size);

			printf("Size of bucket: %llu, Max size: %fGB\n", sizeof(bucket_type), 1.0*(max_items*(sizeof(bucket_type)+sizeof(bucket_type*)))/(1024ULL*1024*1024));

			my_type * host_version = gallatin::utils::get_host_version<my_type>();



			//host_version[0] = default_host_version;

			//host_version->live_bits = gallatin::utils::get_device_version<uint64_t>(host_version->nbits);

			host_version->promote_level = 0;
			host_version->level = n_directory;


			//printf("Live bits %llu, max items: %llu\n", host_version->nbits, max_items);

			//cudaMemset(host_version->live_bits, 0ULL, sizeof(uint64_t)*max_items*2);


			my_type * device_version = gallatin::utils::move_to_device(host_version);

			//init_table_device<my_type><<<1,1>>>(device_version);

			set_table_buckets<my_type><<<(max_items-1)/256+1,256>>>(device_version, min_items);

			// for (int i = 1; i < n_directory; i++){

			// 	//loop through directories and set pointers
			// 	uint64_t loop_n_buckets = min_items << (i-1);

				


			// }


			set_table_pointers<my_type><<<(max_items-1)/256+1,256>>>(device_version, min_items, max_items);

			prealloc_all_buckets<my_type><<<(max_items-1)/256+1,256>>>(device_version, min_items, max_items);

			verify_starting_conditions<my_type><<<(max_items-1)/256+1,256>>>(device_version, max_items);

			cudaDeviceSynchronize();

			printf("Table booted successfully\n");

			return device_version;


		}


		static __host__ void free_on_device(my_type * dev_version){


			clip_ext_buckets<my_type><<<(max_items-1)/256+1, 256>>>(dev_version, max_items);
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

			//make non-global as static.

			return level-1;
			//return cg::invoke_one_broadcast(team, [&] () { return gallatin::utils::ld_acq(&level)-1; });
								

		}


		__device__ bpt pack_together(bucket_type * pointers, uint size){

			return bpt::pack_together(pointers, size);

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


		__device__ bool attach_bucket(bpt bucket, uint64_t position){


			bpt * bucket_address = get_address_of_bucket(position);

			uint64_t result = atomicCAS((unsigned long long int *)bucket_address, 0ULL, (unsigned long long int)bucket);

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


		// __device__ uint64_t get_local_position(uint64_t clipped_hash, uint64_t index){

		// 	if (index == 0) return clipped_hash;

		// 	if (index == 1) return clipped_hash - min_items;

		// 	uint64_t items_at_level_below = min_items + min_items << (index-2);

		// 	return clipped_hash - items_at_level_below;


		// }



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


		//given a bucket, perform a CAS to update load!
		__device__ bool swap_bucket_in_index(uint64_t index, bpt old_bucket, bpt new_bucket){



			bpt * bucket_address = get_address_of_bucket(index);


			return (atomicCAS((unsigned long long int *)bucket_address, (unsigned long long int) old_bucket, (unsigned long long int) new_bucket) == (unsigned long long int) old_bucket);


			

		}

		__device__ void force_bucket_exchange(uint64_t index, bpt new_bucket){

			bpt * bucket_address = get_address_of_bucket(index);

			atomicExch((unsigned long long int *)bucket_address, (unsigned long long int) new_bucket);

		}

		//get the index of a bucket
		__device__ bpt * get_address_of_bucket(uint64_t index){


			// uint64_t directory_index = get_directory_index(index);

			// uint64_t local_position = get_local_position(index, directory_index);

			// bucket_type ** global_read_directory = directory[directory_index];

			// while (global_read_directory == nullptr){

			// 	//printf("Stalling in read of global directory\n");

			// 	//BLEGH
			// 	//global_read_directory = (bucket_type ** ) atomicCAS((unsigned long long int *)&directory[directory_index],0ULL, 0ULL);

			// 	//better if this works.
			// 	global_read_directory = (bucket_type **) gallatin::utils::ld_acq((uint64_t *)&directory[directory_index]);


			// 	#if HT_PRINT
			// 	printf("Looping %lu\n", gallatin::utils::get_tid());
			// 	#endif

			// }


			return &directory[index];

			//return &global_read_directory[local_position];



		}

		//bpt's are only allowed to point to live buckets.
		//verify that the thing referred to at this index is pointing to itself both ways.
		__device__ void verify_pointing_at_self(uint64_t index){

			bpt bucket_pointer = get_bucket_from_index(index);

			auto size = bucket_pointer.get_size();


			if (clip_to_global_level(size, index) != index){
				printf("Bucket points at different index\n");
			}

			// auto clipped_index = clip_to_global_level(size, index);

			// auto bpt_index_bpt = get_bucket_from_index(clipped_index);

			// auto main_ptr = bucket_pointer.get_pointer();

			// auto bucket_size = main_ptr->load_size_atomic_singleton();



		}

		__device__ bpt get_bucket_from_index(uint64_t index, bool load_atomic=false){

			bpt * bucket_ptr = get_address_of_bucket(index);

			return (bpt) gallatin::utils::ld_acq((uint64_t *)bucket_ptr);


		}


		__device__ bpt get_bucket_for_fill(uint64_t index, bool load_atomic=false){



			bpt * bucket_address = get_address_of_bucket(index);



			if (load_atomic && bucket_address[0] == nullptr){

				return (bpt) atomicCAS((unsigned long long int *)bucket_address, 0ULL, 0ULL);

			} else {
				return bucket_address[0];
			}

		}


		//bucket needs to incorporate size.
		__device__ bool insert(Key key, Val val, cg::thread_block_tile<group_size> team){


			uint64_t hash = cooperative_get_hash(key, team);
			uint64_t global_level = cooperative_get_global_level(team);
			uint64_t local_level = global_level;

			

			while (true){

				#if HT_PRINT
				printf("%llu Looping in main\n", gallatin::utils::get_tid());
				#endif

				//force refinement if not valid
				uint64_t bucket_index = clip_to_global_level(local_level, hash);

				// if (team.thread_rank() == 0){
				// 	primary_bucket_wrapper = get_bucket_from_index(bucket_index);
				// }
				// primary_bucket_wrapper = team.shfl(primary_bucket_wrapper, 0);


				bpt primary_bucket_wrapper = cg::invoke_one_broadcast(team, [&] () { return get_bucket_from_index(bucket_index); });

				//base case - bucket should always be 0.
				// if (local_level == 0){

				// 	primary_bucket = get_bucket_from_index(bucket_index, true);

				// 	if (primary_bucket == nullptr){
				// 		printf("Bug setting primary bucket in index %llu\n", bucket_index);
				// 		return false;
				// 	}

				// }

				//printf("Tid %llu Looping on local level %lu\n", gallatin::utils::get_tid(), local_level);


				bucket_type * primary_bucket = primary_bucket_wrapper.get_pointer();

				uint expected_size = primary_bucket_wrapper.get_size();

				//refactor - insert
				//unrolling loop
				//this cannot occur now.
				if (primary_bucket == nullptr){

					printf("This cannot occur.\n");
					continue;

				}

				int insert_slot = primary_bucket->insert(key, val, team);

				auto bucket_size = primary_bucket->load_size_atomic(team);

				//if size is correct we don't care about upper pointer.
				auto local_bucket_index = clip_to_global_level(bucket_size, hash);
				//auto bucket_self_index = clip_to_global_level()

				//what we expected was correct! return.
				if (bucket_size == expected_size){

					//correct behavior.

					if (insert_slot == -1){

						//could not insert, but we have the right bucket.

						if (bucket_size == global_level) return false;


						//do a load and double check.

						#if HT_PRINT
						auto copy_of_primary = get_bucket_from_index(clip_to_global_level(expected_size, hash));

						bucket_type * copy_pointer = copy_of_primary.get_pointer();

						if (copy_pointer != primary_bucket){
							printf("Issue with expected size %llx, %llx, %llu off\n", (uint64_t) copy_pointer, (uint64_t) primary_bucket, (uint64_t) copy_of_primary.get_pointer() - (uint64_t) primary_bucket);
							continue;
						}
						#endif


						maybe_add_new_bucket(local_bucket_index, expected_size+1, primary_bucket, team);

						continue;
					}

					//otherwise correct.

					return true;
				}



				//in this case - we are clearly wrong.
				//unset and update
				if (insert_slot != -1){

					bool rolled_back = cg::invoke_one_broadcast(team, [&] () { return primary_bucket->resetExact(insert_slot, key); });

					//insert succeeded as another thread performed the move for us.
					//other threads will not move for rollback unless exactly one off on size due to race.
					if (!rolled_back) return true;

				}


				//broken update path - node not updated yet?
				//need to generate new path and trace UNTIL match (expected size is ours.)

				//wait to ensure that resize is not in the middle of occurring.
				primary_bucket->wait_on_bucket_promote();

				uint64_t next_index = clip_to_global_level(expected_size+1, hash);

				//verify_pointing_at_self(next_index);


				//new pointer gather
				auto next_pointer_wrapper = get_bucket_from_index(clip_to_global_level(expected_size+1, hash));

				auto new_pack = pack_together(next_pointer_wrapper.get_pointer(), expected_size+1);

				swap_bucket_in_index(clip_to_global_level(global_level, hash), primary_bucket_wrapper, new_pack);

				//otherwise we rolled back. - flense old pointer and update.


				//step one at a time until we find next valid bucket, i.e. one that registers
				// while (true){
				// 	//find next bucket in chain.
				// 	auto correct_bucket_index = clip_to_global_level(expected_size+1, hash);

				// 	bpt new_bucket = get_bucket_from_index(correct_bucket_index);

				// 	uint new_bucket_size = new_bucket.get_pointer()->load_size_atomic(team);

				// 	if (new_bucket_size == expected_size+1){

				// 		bpt new_pack = bpt::pack_together(new_bucket.get_pointer(), expected_size+1);

				// 		swap_bucket_in_index(clip_to_global_level(global_level, hash), primary_bucket_wrapper, new_pack);

				// 		break;

				// 	}

				// }

				// continue;

				// auto correct_bucket_index = clip_to_global_level(expected_size+1, hash);

				// bucket_type * new_bucket = get_bucket_from_index(correct_bucket_index);

				// bpt new_pack = (new)

				// swap_bucket_in_index(clip_to_global_level(global_level, hash), primary_bucket, new_bucket);

				__threadfence();
				continue;

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

						#if HT_PRINT
						printf("Bucket %llx Failed to insert from %d to %llu\n", (unsigned long long int) start, i, moved_keys+keys_below);
						#endif

					}

				}

				//update tracking.
				moved_keys += __popc(ballot_moving);


			}


			team.sync();


			bpt end_wrapped = bpt::pack_together(end, promotion_size);


			//bool bucket_attached = 
			cg::invoke_one(team, [&] () { return force_bucket_exchange(alt_index, end_wrapped); } );

			// if (!bucket_attached){

			// 	printf("Failed to attach!\n");

			// 	asm volatile("trap;");
			// }

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


				// bpt copy_primary_bucket = get_bucket_from_index(index);

				// if (primary_bucket != copy_primary_bucket){

				// 	printf("Maybe add new bucket not matches: %llx != %llx\n", (uint64_t) primary_bucket, (uint64_t) copy_primary_bucket);
				// }


				bucket_type * alt_bucket = (bucket_type *) cg::invoke_one_broadcast(team, [&] () { return atomicExch((unsigned long long int *)&prealloc_directory[alt_index], 0ULL);}); 

				//prealloc_directory[alt_index] = nullptr;

				__threadfence();


				#if HT_PRINT
				if (alt_bucket == nullptr){
					printf("Disaster! Index %llu is already moved\n", alt_index);
				}

				if (alt_bucket->load_size_atomic_singleton() != promotion_size){
					printf("Promotion failure! %u != %u\n", alt_bucket->load_size_atomic_singleton(), promotion_size);
				}

				#endif
				//cg::invoke_one_broadcast(team, [&] () { return get_new_bucket(promotion_size);});


				//bool failed_attach = false;


				// if (team.thread_rank() == 0){

				// 	alt_bucket->stall_lock();


				// 		//printf("Bucket added\n");

				// }

				//cg::invoke_one(team, [&] () { alt_bucket->stall_lock(); });

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

					//alt_bucket->unlock();
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

				auto primary_bucket_wrapper = cg::invoke_one_broadcast(team, [&] () { return get_bucket_from_index(bucket_index, true); });


				auto primary_bucket = primary_bucket_wrapper.get_pointer();

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