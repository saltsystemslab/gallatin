#ifndef SOA_H
#define SOA_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace representations { 

template <typename Key, typename Val, std::size_t Bucket_Size, std::size_t Partition_Size>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  struct_of_arrays {

	private:

	public:

		//using filled_container_type = Container<SmallKey, Val>;

		Key keys[Bucket_Size];
		Val vals[Bucket_Size];


		__device__ int get_fill(cg::thread_block_tile<Partition_Size> insert_tile){


			int fill = 0;

			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){

				bool filled = (keys[i] != get_empty());

				fill += __popc(insert_tile.ballot(filled));

			}

			return fill;

		}

		__device__ inline const Key get_empty(){

			return Key{0};
		}

		//subtract from 0 to cause rollover
		__device__ inline const Key get_tombstone(){
			return get_empty()-1;
		}

		__device__ inline bool contains(cg::thread_block_tile<Partition_Size> insert_tile, Key ext_key){


			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool found = (keys[i] == ext_key);

				if (insert_tile.ballot(found)) return true;

			}

			return false;

		}

		static __device__ inline Key tag(Key ext_key){

			return ext_key;

		}

		__device__ inline bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){



			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				if (keys[i] == get_empty()){
					ballot = true;
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){
						ballot = poggers::helpers::typed_atomic_write(&keys[i], get_empty(), key);

						if (ballot){
							vals[i] = val;
						}
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}


			}

			return false;


		}


		__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key ext_key){

			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				if (keys[i] == ext_key){
					ballot = true;
					//ext_val = vals[i];
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){
						ballot = typed_atomic_write(&keys[i], ext_key, get_tombstone());
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

				if (keys[i] == ext_key){
					ballot = true;
					ext_val = vals[i];
				}

				auto ballot_result = insert_tile.ballot(ballot);

				if (ballot_result){
					ext_val = insert_tile.shfl(ext_val, __ffs(ballot_result)-1);
					return true;
				}



				


			}

			return false;

		}

		

};

//rubber ducky strucky
//this will allow you to set the size of smallKey inside of key_val_pair
//while still allowing all the components to connect nicely.
// template<template <typename, typename> class Container, typename SmallKey> struct dynamic_container {
//     template<typename Key, typename Val>
//     using representation = internal_dynamic_container<Container, Key, Val, SmallKey>;
// };



// template <typename Key, typename Val>
// __device__ void pack_into_pair(key_val_pair<Key, Val, SmallKey> & pair, Key & new_key, Val & new_val ){

// 	pair.set_key(new_key);
// 	pair.set_val(new_val);


// }

// template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
// struct key_val_pair{


// 	//tag bits change based on the #of bytes allocated per block

// 	storage_pair<Key, Val> internal_storage;

// 	//key_val_pair (Key const)

// 	//you only use this when you have vals
// 	key_val_pair(Key const & key, Val const & val): storage_pair<Key, Val>(key, val){}

// 	key_val_pair(Key const & key): storage_pair<Key, Val>(key){}

// 	key_val_pair(){}

// 	__host__ __device__ Key& key{

// 		return internal_storage.key;
// 	}

// 	__host__ __device__ Val& get_val(){

// 		return internal_storage.get_val();
// 	}



// };

// template <typename Key>
// struct key_val_pair<Key, void>{



// };


}

}


#endif //GPU_BLOCK_