#ifndef BUCKET_12_H
#define BUCKET_12_H


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


//TODO: Does not yet support delete FN fix
//need to batch sub_byte_match to be consistent for all sizes.

template <typename Key, typename Val, typename Storage_type, std::size_t Partition_Size, std::size_t Bucket_Size>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  twelve_bucket {

	private:

	public:

		//using filled_container_type = Container<SmallKey, Val>;

		//bits used is 12

		//(12*Bucket_Size-1) / (sizeof(Storage_type)*8) +1 

		Storage_type keys[(12*Bucket_Size-1)/(sizeof(Storage_type)*8)+1];

		//Val vals[Bucket_Size];


		__device__ int get_fill(cg::thread_block_tile<Partition_Size> insert_tile){


			int fill = 0;

			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){

				Storage_type * key_ptr = &keys[0];
				bool filled = !(poggers::helpers::sub_byte_empty(key_ptr, i));

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


				bool found = poggers::helpers::sub_byte_match<Key>(&keys, ext_key, i);

				if (insert_tile.ballot(found)) return true;

			}

			return false;

		}

		__device__ inline bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){



			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				Storage_type * key_ptr = &keys[0];

				if (poggers::helpers::sub_byte_empty(key_ptr, i)){
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
						Storage_type * key_ptr = &keys[0];

						ballot = poggers::helpers::sub_byte_replace<Key>(key_ptr, get_empty(), key, i);
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

				Storage_type * key_ptr = &keys[0];

				if (poggers::helpers::sub_byte_empty(key_ptr, i) || poggers::helpers::sub_byte_match<Key>(key_ptr, get_tombstone(), key, i)){
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
						Storage_type * key_ptr = &keys[0];


						if (poggers::helpers::sub_byte_empty(key_ptr, i)){
							ballot = poggers::helpers::sub_byte_replace<Key>(key_ptr, get_empty(), key, i);
						} else {
							ballot = poggers::helpers::sub_byte_replace<Key>(key_ptr, get_tombstone(), key, i);
						}
						
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

						ballot = poggers::helpers::sub_byte_replace<Key>(&keys, ext_key, get_tombstone(), i);
						//ballot = typed_atomic_write(&keys[i], ext_key, get_tombstone());
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}

				


			}

			return false;



		}

		__device__ void full_reset(cg::thread_block_tile<Partition_Size> insert_tile){

			for (int i = insert_tile.thread_rank(); i < (12*Bucket_Size-1)/(sizeof(Storage_type)*8)+1; i++){

				keys[i] = 0;

			}
		}

		__device__ inline bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key ext_key, Val & ext_val){


			for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+= Partition_Size){


				bool ballot = false;

				// if (keys[i] == ext_key){
				// 	ballot = true;
				// 	ext_val = 0;
				// }

				Storage_type * key_ptr = &keys[0];

				if (poggers::helpers::sub_byte_match<Key>(key_ptr, ext_key, i)){
					ballot = true;
					ext_val = 0;
				}

				auto ballot_result = insert_tile.ballot(ballot);

				if (ballot_result){
					//ext_val = insert_tile.shfl(ext_val, __ffs(ballot_result)-1);
					ext_val = 0;
					return true;
				}



				


			}

			return false;

		}


		static Key tag(Key key){

		}
		

};

//rubber ducky strucky
//this will allow you to set the size of smallKey inside of key_val_pair
//while still allowing all the components to connect nicely.
template<typename Internal_Storage> struct wrapper_half_bucket {
    template<typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size>
    using representation = twelve_bucket<Key, Val, Internal_Storage, Partition_Size, Bucket_Size>;
};



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