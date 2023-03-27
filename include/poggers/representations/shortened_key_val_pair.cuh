#ifndef SHORTENED_KEY_VAL_PAIR_H 
#define SHORTENED_KEY_VAL_PAIR_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace representations { 

template <typename Key, typename Val, typename SmallKey>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  shortened_bitmask_key_val_pair {

	private:

	public:

		__host__ __device__ static const SmallKey get_empty(){ return SmallKey{0}; }


		static __host__ __device__ inline const SmallKey get_smallKey(Key ext_key){ return SmallKey{ext_key}; }

		SmallKey key;

		Val val;

		__host__ __device__ shortened_bitmask_key_val_pair(){}

		//constructor
		__host__ __device__ shortened_bitmask_key_val_pair (Key const & key, Val const & val)
		: key(get_smallKey(key)), val(val){}

		__device__ __inline__ bool atomic_swap(Key const ext_key, Val const ext_val){
			if (poggers::helpers::typed_atomic_write(&key, get_empty(), get_smallKey(ext_key))){

				val = ext_val;

				return true;

			}

			return false;

		}

		__device__ __inline__ bool atomic_swap_tombstone(Key const ext_key, Val const ext_val){
			if (poggers::helpers::typed_atomic_write(&key, get_tombstone(), get_smallKey(ext_key))){

				val = ext_val;

				return true;

			}

			return false;

		}

		static __device__ inline Key tag(Key ext_key){

			return get_smallKey(ext_key);

		}


		__host__ __device__ inline bool is_empty(){
			return (key == get_empty());
		}

		__host__ __device__ inline bool contains(Key ext_key){
			return (key == get_smallKey(ext_key));
		}

		__host__ __device__ inline bool contains_tombstone(){
			return (key == get_tombstone());
		}

		__host__ __device__ inline Val get_val(Key ext_key){
			return val;
		}

		__host__ __device__ inline void reset(){
			key = get_empty();
		}

		__host__ __device__ inline static const SmallKey get_tombstone(){
			return get_empty()-1;
		}

		__device__ __inline__ bool atomic_reset(Key const ext_key){

			if (poggers::helpers::typed_atomic_write(&key, get_smallKey(ext_key), get_tombstone())){
				return true;
			}

			return false;

		}

		

};

//rubber ducky strucky
//this will allow you to set the size of smallKey inside of key_val_pair
//while still allowing all the components to connect nicely.
template<typename SmallKey> struct shortened_key_val_wrapper {
    template<typename Key, typename Val>
    using key_val_pair = shortened_bitmask_key_val_pair<Key, Val, SmallKey>;
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


template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator<(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){


	return A.key < B.key;

}

template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator<=(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){

	return A.key <= B.key;

}

template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator>=(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){

	return A.key >= B.key;

}

template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator==(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){

	return A.key == B.key;

}

template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator!=(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){

	return A.key != B.key;

}

template <typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator>(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, shortened_bitmask_key_val_pair<Key, Val, SmallKey> B){

	return A.key > B.key;

}


template <typename Key, typename Val, typename SmallKey>
__host__ __device__ Key operator/(shortened_bitmask_key_val_pair<Key, Val, SmallKey> A, uint64_t other){

	return A.key / other;
}

}

}


#endif //GPU_BLOCK_