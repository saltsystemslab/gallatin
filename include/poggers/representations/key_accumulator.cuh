#ifndef KEY_ACC_H 
#define KEY_ACC_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


namespace poggers {


namespace representations { 

template <typename Key, typename Val>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  key_val_pair {

	private:
		

	public:

		__host__ __device__ static inline const Key get_empty(){ return Key{0}; }

		Key key;

		Val val;

		__host__ __device__ key_val_pair(){}

		//constructor
		__host__ __device__ key_val_pair (Key const & key, Val const & val)
		: key(key), val(val){}

		__device__ __inline__ bool atomic_swap(Key const ext_key, Val const ext_val){

			Key result = poggers::helpers::typed_atomic_CAS(&key, get_empty(), ext_key);

			if (result == get_empty() || result == ext_key){

				//increment
				if constexpr(sizeof(Val) == 32){
					atomicIncrement(&Val, maxVal());
					return true;
				}

				else if (constexpr((sizeof(Val) == 16) && (sizof(key) == 16))){

					uint32_t cap = (ext_key << 16) | maxVal();
					atomicIncrement(&Val, cap);
					return true;
				} else {


					abort();
				}

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

		__host__ __device__ inline Val get_val(Key ext_key){
			return val;
		}


		static __device__ inline Key tag(Key ext_key){

			return ext_key;

		}

		//cycle two down to prevent rollover cases
		__host__ __device__ static const Val maxVal(){

			return Val{0}-2;
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
// __device__ void pack_into_pair(key_val_pair<Key, Val> & pair, Key & new_key, Val & new_val ){

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
__host__ __device__ bool operator<(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){


	return A.key < B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator<=(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){

	return A.key <= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>=(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){

	return A.key >= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator==(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){

	return A.key == B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator!=(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){

	return A.key != B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>(key_val_pair<Key, Val> A, key_val_pair<Key, Val> B){

	return A.key > B.key;

}


template <typename Key, typename Val>
__host__ __device__ Key operator/(key_val_pair<Key, Val >A, uint64_t other){

	return A.key / other;
}

}

}


#endif //GPU_BLOCK_