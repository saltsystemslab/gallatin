#ifndef PACKED_BUCKET_H 
#define PACKED_BUCKET_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

//bucketed internal dynamic container stores a list of 
//some variant of key_val pairs
//equivalent to a 12-bit bucket but for sizes that fit neatly within a byte.

namespace poggers {


namespace representations { 

template <template <typename, typename> class Container, typename Key, typename Val, std::size_t Partition_Size, size_t Bucket_Size>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  bucketed_internal_dynamic_container {

	private:

	public:

		using filled_container_type = Container<Key, Val>;

		filled_container_type storage [Bucket_Size];


		__device__ int get_fill(cg::thread_block_tile<Partition_Size> insert_tile){

			int fill = 0;

			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){

				// //Storage_type * key_ptr = &keys[0];
				// if (threadIdx.x+blockIdx.x*blockDim.x == 2){


				// printf("%d: is_empty: %d, is full: %d tombstone %x contained: %x\n", i, storage[i].is_empty(), storage[i].contains(storage[i].get_tombstone()), storage[i].get_tombstone(), storage[i]);
				// }

				bool filled = !(storage[i].is_empty() || storage[i].contains_tombstone());

				fill += __popc(insert_tile.ballot(filled));

			}

			return fill;


		}

		__device__ inline bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){

				bool ballot = false;

				//Storage_type * key_ptr = &keys[0];

				if (storage[i].is_empty()){
				//if (poggers::helpers::sub_byte_match<Key>(key_ptr, get_empty(), i)){
					ballot = true;
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){

						//printf("Inserting\n");

						//poggers::helpers::sub_byte_contains<Storage_type, Key, 12, Bucket_Size>(&keys, i)
						//Storage_type * key_ptr = &keys[0];

						ballot = storage[i].atomic_swap(key, val);

						//ballot = poggers::helpers::sub_byte_replace<Key>(key_ptr, get_empty(), key, i);
						//ballot = poggers::helpers::typed_atomic_write(&keys[i], get_empty(), key);

						// if (ballot){
						// 	vals[i] = val;
						// }
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}


			}

			return false;


		}

		__device__ inline bool insert_delete(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){

				bool ballot = false;

				//Storage_type * key_ptr = &keys[0];

				if (storage[i].is_empty() || storage[i].contains_tombstone()){
				//if (poggers::helpers::sub_byte_match<Key>(key_ptr, get_empty(), i)){
					ballot = true;
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){

						//printf("Inserting\n");

						//poggers::helpers::sub_byte_contains<Storage_type, Key, 12, Bucket_Size>(&keys, i)
						//Storage_type * key_ptr = &keys[0];

						if (storage[i].is_empty()){

							ballot = storage[i].atomic_swap(key, val);

						} else {

							ballot = storage[i].atomic_swap_tombstone(key, val);


							// if (!ballot && storage[i].contains_tombstone()){
							// 	printf("Bug in swapping\n");
							// }
						}

						

						//ballot = poggers::helpers::sub_byte_replace<Key>(key_ptr, get_empty(), key, i);
						//ballot = poggers::helpers::typed_atomic_write(&keys[i], get_empty(), key);

						// if (ballot){
						// 	vals[i] = val;
						// }
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}


			}

			return false;


		}

		__device__ inline bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key ext_key, Val & ext_val){


			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				// if (keys[i] == ext_key){
				// 	ballot = true;
				// 	ext_val = 0;
				// }

				//Storage_type * key_ptr = &keys[0];

				if (storage[i].contains(ext_key)){
					ballot = true;
					ext_val = storage[i].get_val(ext_key);
				}

				auto ballot_result = insert_tile.ballot(ballot);

				if (ballot_result){
					ext_val = insert_tile.shfl(ext_val, __ffs(ballot_result)-1);
					//ext_val = 0;
					return true;
				}



				


			}

			return false;

		}

		static __device__ inline Key tag(Key ext_key){

			return filled_container_type::tag(ext_key);

		}

		__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key ext_key){


			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				if (storage[i].contains(ext_key)){
					ballot = true;
					//ext_val = vals[i];
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){

						ballot = storage[i].atomic_reset(ext_key);
						//ballot = true;
						//poggers::helpers::sub_byte_replace<Key>(&keys, ext_key, get_tombstone(), i);
						//ballot = typed_atomic_write(&keys[i], ext_key, get_tombstone());
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}

				


			}

			return false;



		}


		__host__ __device__ bucketed_internal_dynamic_container(){}

		

};

//rubber ducky strucky
//this will allow you to set the size of smallKey inside of key_val_pair
//while still allowing all the components to connect nicely.
template<template <typename, typename> class Container> struct dynamic_bucket_container {
    template<typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size>
    using representation = bucketed_internal_dynamic_container<Container, Key, Val, Partition_Size, Bucket_Size>;
};



// template <typename Key, typename Val>
// __device__ void pack_into_pair(key_val_pair<Key, Val, SmallKey> & pair, Key & new_key, Val & new_val ){

// 	pair.set_key(new_key);
// 	pair.set_val(new_val);





}

}


#endif //GPU_BLOCK_