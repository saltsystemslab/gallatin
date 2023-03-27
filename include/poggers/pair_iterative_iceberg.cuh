#ifndef SIMPLE_TABLE_H 
#define SIMPLE_TABLE_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/metadata.cuh"
#include "include/ht_helpers.cuh"
//#include "include/key_val_pair.cuh"
//#include "include/templated_block.cuh"
#include "include/hashutil.cuh"
//#include "include/templated_sorting_funcs.cuh"
#include <stdio.h>
#include <assert.h>

//thrust stuff
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "include/key_val_pair.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs

// template <typename Tag_type>
// __device__ bool assert_sorted(Tag_type * tags, int nitems){


// 	if (nitems < 1) return true;

// 	Tag_type smallest = tags[0];

// 	for (int i=1; i< nitems; i++){

// 		if (tags[i] < smallest) return false;

// 		smallest = tags[i];
// 	}

// 	return true;

// }

//specialized atomic_CAS
enum query_return_type: short{
	ZERO_FOUND,
	NOT_FOUND,
	FOUND
};







template <typename Key, typename Val, std::size_t Primary_bucket_size, std::size_t Secondary_bucket_size, std::size_t Partition_size, bool Has_backyard>
struct __attribute__ ((__packed__)) static_size_iceberg_table {


	//tag bits change based on the #of bytes allocated per block

	
	//typedef key_val_pair<Key> Key;

	uint64_t primary_nitems;

	uint64_t secondary_nitems;

	key_val_pair<Key, Val> * primary_items;

	key_val_pair<Key, Val> * secondary_items;


	//space for backyard here
	//probs a tiny has table? need to measure variance before working on this.



	//three hashes using two murmurhashes - should be no more computationally complex than the hashing used by WarpCore

	__device__ __inline__ uint64_t generate_slot_primary_bucket(uint64_t primary_slot, uint64_t secondary_slot){

		uint64_t preadjusted_slot = (primary_slot + secondary_slot) % primary_nitems;

		#if DEBUG_ASSERTS

		assert (preadjusted_slot < primary_nitems);

		#endif

		return (preadjusted_slot - preadjusted_slot % Primary_bucket_size);
	}

	__device__ __inline__ uint64_t generate_slots_secondary_bucket_1(uint64_t primary_slot, uint64_t secondary_slot){

		uint64_t preadjusted_slot = (primary_slot + secondary_slot*2) % secondary_nitems;

		#if DEBUG_ASSERTS

		assert (preadjusted_slot < secondary_nitems);

		#endif

		return (preadjusted_slot - preadjusted_slot % Secondary_bucket_size);
	}

	__device__ __inline__ uint64_t generate_slots_secondary_bucket_2(uint64_t primary_slot, uint64_t secondary_slot){

		uint64_t preadjusted_slot = (primary_slot + secondary_slot*3) % secondary_nitems;

		#if DEBUG_ASSERTS

		assert (preadjusted_slot < secondary_nitems);

		#endif

		return (preadjusted_slot - preadjusted_slot % Secondary_bucket_size);
	}


	__device__ __inline__ bool query_key_secondary_bucket(cg::thread_block_tile<Partition_size> group, Key query_key, uint64_t primary_bucket, uint64_t secondary_bucket){


		int lane = group.thread_rank();

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){

			Key table_key = secondary_items[primary_bucket +i].key;

			const bool hit = (table_key == query_key);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;

		}

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){
			Key table_key = secondary_items[secondary_bucket + i].key;

			const bool hit = (table_key == query_key);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;
		}

		return false;
	}


	__device__ __inline__ bool alternate_insert_key_into_secondary_bucket(cg::thread_block_tile<Partition_size> group, Key insert_key, uint64_t primary_bucket, uint64_t secondary_bucket){

	}


	__device__ __inline__ bool insert_key_into_secondary_bucket(cg::thread_block_tile<Partition_size> group, Key insert_key, Val insert_val, uint64_t primary_bucket, uint64_t secondary_bucket){


		int lane  = group.thread_rank();


		//these should be synced among threads and correspond to the individual items that are being queried
		//double check me in cuda-gdb [TODO]
		uint32_t primary_ballots = 0;


		uint32_t secondary_ballots = 0;

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){

			Key table_key = secondary_items[primary_bucket + i].key;

			const bool hit = (table_key == insert_key);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;

			auto empty_mask = group.ballot(table_key == 0);

			//push into ballots
			primary_ballots ^= empty_mask << (Partition_size*(i / Partition_size));


		}

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){

			Key table_key = secondary_items[secondary_bucket + i].key;

			const bool hit = (table_key == insert_key);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;

			auto empty_mask = group.ballot(table_key == 0);

			//push into ballots
			secondary_ballots ^= empty_mask << (Partition_size*(i / Partition_size));


		}

		if (__popc(primary_ballots) > __popc(secondary_ballots)){


			//swap
			uint32_t temp_ballot = primary_ballots;
			primary_ballots = secondary_ballots;
			secondary_ballots = temp_ballot;

			uint64_t temp_bucket = primary_bucket;
			uint64_t primary_bucket = secondary_bucket;
			uint64_t secondary_bucket = temp_bucket;


		}

		//attempt insert into primary_bucket
		//no need to attempt secondary read as we already

		while (primary_ballots){

		
			const auto leader = __ffs(primary_ballots) - 1;

			bool duplicate = false;

			bool success = false;

			if (leader % Partition_size == lane){

				const auto old = helpers::typed_atomic_write( (Key *) &secondary_items[primary_bucket+leader].key, (Key) 0, insert_key);

				success = (old == 0);
				duplicate = (old == insert_key);

				if (success){
					secondary_items[primary_bucket+leader].val = insert_val;
				}


			}

			if (group.any(duplicate)){
				return true;
			}

			if (group.any(success)){
				return true;
			}

			primary_ballots ^= 1UL << leader;
	
		}

		return false;


		//attempt insert into primary bucket;

	}

	__device__ __inline__ bool insert_key_into_primary_bucket(cg::thread_block_tile<Partition_size> group, Key insert_key, Val insert_val, uint64_t canonical_slot){


		int lane = group.thread_rank();


		for (int i = lane; i < Primary_bucket_size; i+=Partition_size){




			Key table_key = primary_items[canonical_slot + i].key;

			const bool hit = (table_key == insert_key);

			const auto hit_mask = group.ballot(hit);

			if (hit_mask){


				return true;
			}

			auto empty_mask = group.ballot(table_key == 0);

			bool success = false;
			bool duplicate = false;

			while(empty_mask){

				const auto leader = __ffs(empty_mask) -1;

				if (lane == leader){

					const auto old = helpers::typed_atomic_write(&primary_items[canonical_slot+i].key, table_key, insert_key);

					success = (old == table_key);
					duplicate = (old == insert_key);

					if (success) primary_items[canonical_slot+i].val = insert_val;
				}

				if (group.any(duplicate)){
					return true;
				}

				if (group.any(success)){
					return true;
				}

				empty_mask ^= 1UL << leader;
			}


		}

		return false;


	}

	__device__ __inline__ bool insert_pair_into_cache_line(cg::thread_block_tile<Partition_size> group, key_val_pair<Key,Val> * cache_to_load, key_val_pair<Key,Val> insert_pair){


		int lane = group.thread_rank();





		Key table_key = cache_to_load[lane].key;

		const bool hit = (table_key == insert_pair.key);

		const auto hit_mask = group.ballot(hit);

		if (hit_mask){


			return true;
		}

		auto empty_mask = group.ballot(table_key == 0);

		bool success = false;
		bool duplicate = false;

		while(empty_mask){

			const auto leader = __ffs(empty_mask) -1;

			if (lane == leader){

				const auto old = helpers::typed_atomic_write(&cache_to_load[lane].key, table_key, insert_pair.key);

				success = (old == table_key);
				duplicate = (old == insert_pair.key);

				if (success) cache_to_load[lane].val = insert_pair.val;
			}

			if (group.any(duplicate)){
				return true;
			}

			if (group.any(success)){
				return true;
			}

			empty_mask ^= 1UL << leader;
		}



		return false;


	}


	__device__ __inline__ bool insert_pair_into_primary_bucket_cache_algined(cg::thread_block_tile<Partition_size> group, const key_val_pair<Key,Val> pair, uint64_t canonical_slot){


		for (int i = 0; i < Primary_bucket_size/8; i++){

			key_val_pair<Key, Val> * cache_line = primary_items+canonical_slot+i*8;

			if(insert_pair_into_cache_line(group, cache_line, pair)){
				return true;
			}
		}

		return false;
	}



	__device__ __inline__ bool insert_pair_into_primary_bucket(cg::thread_block_tile<Partition_size> group, key_val_pair<Key,Val> pair, uint64_t canonical_slot){


		int lane = group.thread_rank();


		for (int i = lane; i < Primary_bucket_size; i+=Partition_size){




			Key table_key = primary_items[canonical_slot + i].key;

			const bool hit = (table_key == pair.key);

			const auto hit_mask = group.ballot(hit);

			if (hit_mask){


				return true;
			}

			auto empty_mask = group.ballot(table_key == 0);

			bool success = false;
			bool duplicate = false;

			while(empty_mask){

				const auto leader = __ffs(empty_mask) -1;

				if (lane == leader){

					const auto old = helpers::typed_atomic_write(&primary_items[canonical_slot+i].key, table_key, pair.key);

					success = (old == table_key);
					duplicate = (old == pair.key);

					if (success) primary_items[canonical_slot+i].val = pair.val;
				}

				if (group.any(duplicate)){
					return true;
				}

				if (group.any(success)){
					return true;
				}

				empty_mask ^= 1UL << leader;
			}


		}

		return false;


	}


	__device__ __inline__ bool insert_pair_into_secondary_bucket(cg::thread_block_tile<Partition_size> group, key_val_pair<Key, Val> pair, uint64_t primary_bucket, uint64_t secondary_bucket){


		int lane  = group.thread_rank();


		//these should be synced among threads and correspond to the individual items that are being queried
		//double check me in cuda-gdb [TODO]
		uint32_t primary_ballots = 0;


		uint32_t secondary_ballots = 0;

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){

			key_val_pair<Key, Val> table_key = secondary_items[primary_bucket + i];

			const bool hit = (table_key == pair);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;

			auto empty_mask = group.ballot(table_key.key == 0);

			//push into ballots
			primary_ballots ^= empty_mask << (Partition_size*(i / Partition_size));


		}

		for (int i = lane; i < Secondary_bucket_size; i+=Partition_size){

			key_val_pair<Key, Val> table_key = secondary_items[secondary_bucket + i];

			const bool hit = (table_key == pair);
			const auto hit_mask = group.ballot(hit);

			if (hit_mask) return true;

			auto empty_mask = group.ballot(table_key.key == 0);

			//push into ballots
			secondary_ballots ^= empty_mask << (Partition_size*(i / Partition_size));


		}

		if (__popc(primary_ballots) > __popc(secondary_ballots)){


			//swap
			uint32_t temp_ballot = primary_ballots;
			primary_ballots = secondary_ballots;
			secondary_ballots = temp_ballot;

			uint64_t temp_bucket = primary_bucket;
			uint64_t primary_bucket = secondary_bucket;
			uint64_t secondary_bucket = temp_bucket;


		}

		//attempt insert into primary_bucket
		//no need to attempt secondary read as we already

		while (primary_ballots){

		
			const auto leader = __ffs(primary_ballots) - 1;

			bool duplicate = false;

			bool success = false;

			if (leader % Partition_size == lane){

				const auto old = helpers::typed_atomic_write( (Key *) &secondary_items[primary_bucket+leader].key, (Key) 0, pair.key);

				success = (old == 0);
				duplicate = (old == pair.key);

				if (success){
					secondary_items[primary_bucket+leader] = pair;
				}


			}

			if (group.any(duplicate)){
				return true;
			}

			if (group.any(success)){
				return true;
			}

			primary_ballots ^= 1UL << leader;
	
		}

		return false;


		//attempt insert into primary bucket;

	}



	__device__ __inline__ query_return_type query_exists_primary_bucket(cg::thread_block_tile<Partition_size> group, Key query_key, uint64_t canonical_slot){


		int lane = group.thread_rank();

		for (int i = lane; i < Primary_bucket_size; i+= Partition_size){


			Key table_key = primary_items[canonical_slot + i].key;

			const bool hit = (table_key == query_key);

			auto hit_mask = group.ballot(hit);

			if (hit_mask){
				return FOUND;
			}

			auto empty_mask = group.ballot((table_key == 0));

			if (empty_mask){
				return ZERO_FOUND;
			}
		}


		return NOT_FOUND;

	}




	__device__ bool query_key(cg::thread_block_tile<Partition_size> group, Key key){

		uint64_t primary_hash = get_primary_hash_from_key(key) ;

		uint64_t secondary_hash = get_secondary_hash_from_key(key);

		query_return_type primary_check = query_exists_primary_bucket(group, key, generate_slot_primary_bucket(primary_hash, secondary_hash));


		if (primary_check == ZERO_FOUND){
			return false;
		}

		if (primary_check == NOT_FOUND){

			if (!query_key_secondary_bucket(group, key, generate_slots_secondary_bucket_1(primary_hash, secondary_hash), generate_slots_secondary_bucket_2(primary_hash, secondary_hash))){


				return false;

			}


		}


		return true;
	}


	//for now just assume everything is a type that can be CAS'd
	__device__ bool insert_key(cg::thread_block_tile<Partition_size> group, Key key, Val val){




		uint64_t primary_hash = get_primary_hash_from_key(key) ;

		uint64_t secondary_hash = get_secondary_hash_from_key(key);

		if (!insert_key_into_primary_bucket(group, key, val, generate_slot_primary_bucket(primary_hash, secondary_hash))){


			//space to attempt deeper inserts

			if (!insert_key_into_secondary_bucket(group, key, val, generate_slots_secondary_bucket_1(primary_hash, secondary_hash), generate_slots_secondary_bucket_2(primary_hash, secondary_hash))){


				return false;
			}


		

		}

		return true;

	}


		//for now just assume everything is a type that can be CAS'd
	__device__ bool insert_key_pair(cg::thread_block_tile<Partition_size> group, const key_val_pair<Key, Val> pair){




		uint64_t primary_hash = get_primary_hash_from_key(pair.key) ;

		uint64_t secondary_hash = get_secondary_hash_from_key(pair.key);

		if (!insert_pair_into_primary_bucket_cache_algined(group, pair, generate_slot_primary_bucket(primary_hash, secondary_hash))){


			//space to attempt deeper inserts

			//if (!insert_pair_into_secondary_bucket(group, pair, generate_slots_secondary_bucket_1(primary_hash, secondary_hash), generate_slots_secondary_bucket_2(primary_hash, secondary_hash))){


				return false;
			//}

		

		}

		return true;

	}


	// __device__ bool insert_key_pair(cg::thread_block_tile<Partition_size> group, key_val_pair<Key, Val> pair){


	// 	uint64_t primary_hash = get_primary_hash_from_key(pair.key);

	// 	uint64_t secondary_hash = get_secondary_hash_from_key(pair.key);

	// 	if (!insert_key_into_primary_bucket)


	// 	//TODO


	// }



	

	__device__ void insert_keys_device_side_misses(Key * keys, Val * vals, uint64_t nkeys, uint64_t * misses){


		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		

		if(!insert_key(insert_tile, keys[my_group], vals[my_group])){

			//insert_key(insert_tile, keys[my_group]);


			 if (insert_tile.thread_rank() == 0){

				atomicAdd((unsigned long long int *) misses, 1ULL);

			}


		}



	}

	__device__ void insert_keys_device_side(Key * keys, Val * vals,  uint64_t nkeys){


		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		insert_key(insert_tile, keys[my_group], vals[my_group]);



	}

	__device__ void insert_pairs_device_side(const key_val_pair<Key, Val> * pairs,  uint64_t nkeys){


		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		insert_key_pair(insert_tile, pairs[my_group]);



	}

	__device__ void insert_pairs_device_side_misses(const key_val_pair<Key, Val> * pairs, uint64_t nkeys, uint64_t * misses){

	

		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		if(!insert_key_pair(insert_tile, pairs[my_group])){

			//insert_key(insert_tile, keys[my_group]);


			 if (insert_tile.thread_rank() == 0){

				atomicAdd((unsigned long long int *) misses, 1ULL);

			}


		}


	}

	__device__ void query_keys_device_side(const Key * keys,  uint64_t nkeys){

	

		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		query_key(insert_tile, keys[my_group]);


	}

	__device__ void query_keys_device_side_misses(Key * keys,  uint64_t nkeys, uint64_t * misses){

		

		auto thread_block = cg::this_thread_block();

		cg::thread_block_tile<Partition_size> insert_tile = cg::tiled_partition<Partition_size>(thread_block);

		uint64_t my_group = blockIdx.x*insert_tile.meta_group_size() + insert_tile.meta_group_rank();

		if (my_group >= nkeys) return;

		if (!query_key(insert_tile, keys[my_group]) && insert_tile.thread_rank() == 0){

			atomicAdd((unsigned long long int *) misses, 1ULL);

		}


	}




	//device functions
	__device__ uint64_t get_primary_hash_from_key(Key key){

		
		uint64_t ret_key = MurmurHash64A(((void *)&key), sizeof(Key), 42);

		return ret_key;

	}

	__device__ uint64_t get_secondary_hash_from_key(Key key){

		
		uint64_t ret_key = MurmurHash64A(((void *)&key), sizeof(Key), 1999);

		return ret_key;


	}


	__host__ void bulk_insert(Key * keys, Val * vals, uint64_t nvals){


		helpers::bulk_insert_kernel<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key, Val><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, keys, vals, nvals);

	}

	__host__ void bulk_insert_misses(Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){


		helpers::bulk_insert_kernel_misses<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key, Val><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, keys, vals, nvals, misses);


	}

	__host__ void bulk_pair_insert(key_val_pair<Key, Val> * pairs, uint64_t nvals){


		helpers::bulk_pair_insert_kernel<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key, Val><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, pairs, nvals);

	}

	__host__ void bulk_pair_insert_misses(key_val_pair<Key, Val> * pairs, uint64_t nvals, uint64_t * misses){


		helpers::bulk_pair_insert_kernel_misses<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key, Val><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, pairs, nvals, misses);
		

	}



	//Queries
	__host__ void bulk_query(Key * keys, uint64_t nvals){

		helpers::bulk_query_kernel<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, keys, nvals);


	}

	__host__ void bulk_query_misses(Key * keys, uint64_t nvals, uint64_t * misses){

		helpers::bulk_query_kernel_misses<static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>, Key><<<(Partition_size*nvals -1)/BLOCK_SIZE +1, BLOCK_SIZE>>>(this, keys, nvals, misses);


	}

};



template <typename Key, typename Val, std::size_t Primary_bucket_size, std::size_t Secondary_bucket_size, std::size_t Partition_size, bool Has_backyard>
__host__ void free_ht(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard> * vqf){


	static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard> * host_vqf;


	cudaMallocHost((void **)& host_vqf, sizeof(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>));

	cudaMemcpy(host_vqf, vqf, sizeof(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>), cudaMemcpyDeviceToHost);

	cudaFree(vqf);

	cudaFree(host_vqf->primary_items);

	cudaFree(host_vqf->secondary_items);

	cudaFreeHost(host_vqf);




}


//TODO - put this in a freaking constructor.
template <typename Key, typename Val, std::size_t Primary_bucket_size, std::size_t Secondary_bucket_size, std::size_t Partition_size, bool Has_backyard>
__host__ static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard> * build_ht(uint64_t nitems){


	static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard> * host_vqf;

	cudaMallocHost((void **)&host_vqf, sizeof(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>));

	const int primary_bucket_size_int = Primary_bucket_size;
	const int secondary_bucket_size_int = Secondary_bucket_size;

	key_val_pair<Key, Val> * primary_items;

	key_val_pair<Key, Val> * secondary_items;

	uint64_t primary_nitems = ((nitems -1) /  primary_bucket_size_int + 1)* primary_bucket_size_int;

	//printf("")
	printf("Num blocks %llu\n",primary_nitems/Primary_bucket_size);

	//im guestimating 10% of the size of the main?
	uint64_t secondary_nitems = ((nitems-1)/primary_bucket_size_int +1)*secondary_bucket_size_int;

	cudaMalloc((void **)&primary_items, primary_nitems*sizeof(key_val_pair<Key, Val>));

	cudaMalloc((void **)&secondary_items, secondary_nitems*sizeof(key_val_pair<Key, Val>));

	host_vqf->primary_items = primary_items;

	host_vqf->secondary_items = secondary_items;




	host_vqf->primary_nitems = primary_nitems;
	host_vqf->secondary_nitems = secondary_nitems;


	static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard> * dev_vqf;


	cudaMalloc((void **)& dev_vqf, sizeof(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>));

	cudaMemcpy(dev_vqf, host_vqf, sizeof(static_size_iceberg_table<Key, Val, Primary_bucket_size, Secondary_bucket_size, Partition_size, Has_backyard>), cudaMemcpyHostToDevice);

	cudaFreeHost(host_vqf);



	return dev_vqf;

}



#endif //GPU_BLOCK_