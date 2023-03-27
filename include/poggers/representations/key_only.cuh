#ifndef KEY_CONTAINER_H 
#define KEY_CONTAINER_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };


namespace poggers {


namespace representations { 

template <typename Key, typename Val>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  key_container {

	private:
		

	public:

		__host__ __device__ static inline const Key get_empty(){ return Key{0}; }

		Key key;

		__host__ __device__ key_container(){}

		//constructor
		__host__ __device__ key_container (Key const & key, Val const & val)
		: key(key){}

		__device__ __inline__ bool atomic_swap(Key const ext_key, Val const ext_val){
			if (poggers::helpers::typed_atomic_write(&key, get_empty(), ext_key)){

				return true;

			}

			return false;

		}

		__host__ __device__ inline bool is_empty(){
			return (key == get_empty());
		}

		__host__ __device__ inline static const Key get_tombstone(){
			return get_empty()-1;
		}

		__host__ __device__ inline bool contains(Key ext_key){
			return (key == ext_key);
		}

		__host__ __device__ inline bool contains_tombstone(){
			return contains(get_tombstone());
		}

		static __device__ inline Key tag(Key ext_key){

			return ext_key;

		}

		__host__ __device__ const inline Val get_val(Key ext_key){
			return (Val) 0;
		}

		__host__ __device__ inline void reset(){

			key = get_empty();

		}

		__device__ __inline__ bool atomic_reset(Key const ext_key){

			if (poggers::helpers::typed_atomic_write(&key, ext_key, get_tombstone())){
				return true;
			}

			return false;

		}

		__device__ __inline__ bool atomic_swap_tombstone(Key const ext_key, Val const ext_val){

			if (poggers::helpers::typed_atomic_write(&key, ext_key, get_tombstone())){

				//val = ext_val;

				return true;

			}

			return false;

		}


		__device__ inline bool atomic_insert(Key const ext_key, Val const ext_val){

			if (is_empty()){
				return atomic_swap(ext_key, ext_val);
			} else {
				return atomic_swap_tombstone(ext_key, ext_val);
			}

		}

		__device__ inline bool is_empty_or_tombstone(){
			return is_empty() || contains_tombstone();
		}

		

};


// template <typename Key, typename Val>
// __device__ void pack_into_pair(key_container<Key, Val> & pair, Key & new_key, Val & new_val ){

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


template <typename Key, typename Val>
__host__ __device__ bool operator<(key_container<Key, Val> A, key_container<Key, Val> B){


	return A.key < B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator<=(key_container<Key, Val> A, key_container<Key, Val> B){

	return A.key <= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>=(key_container<Key, Val> A, key_container<Key, Val> B){

	return A.key >= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator==(key_container<Key, Val> A, key_container<Key, Val> B){

	return A.key == B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator!=(key_container<Key, Val> A, key_container<Key, Val> B){

	return A.key != B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>(key_container<Key, Val> A, key_container<Key, Val> B){

	return A.key > B.key;

}


template <typename Key, typename Val>
__host__ __device__ Key operator/(key_container<Key, Val >A, uint64_t other){

	return A.key / other;
}

}

}


#endif //GPU_BLOCK_