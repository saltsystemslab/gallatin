#ifndef DYNAMIC_CONTAINER_H 
#define DYNAMIC_CONTAINER_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace representations { 

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  internal_dynamic_container {

	private:

	public:

		using filled_container_type = Container<SmallKey, Val>;

		filled_container_type storage;

		static __host__ __device__ inline const SmallKey get_smallKey(Key ext_key){ 


			SmallKey smaller_version = (SmallKey) ext_key; 

			if (smaller_version == get_empty()){
				smaller_version += 1;
			}

			if (smaller_version == get_tombstone()){
				smaller_version -= 1;
			}

			return smaller_version;

		}

		static __device__ inline Key tag(Key ext_key){
			return get_smallKey(ext_key);
		}

		__host__ __device__ static const SmallKey get_empty(){ return filled_container_type::get_empty(); }



		__host__ __device__ internal_dynamic_container(){}

		//constructor
		__host__ __device__ internal_dynamic_container(Key const & key, Val const & val)
		: storage(get_smallKey(key), val){}

		__device__ __inline__ bool atomic_swap(Key const ext_key, Val const ext_val){
	
			return storage.atomic_swap(get_smallKey(ext_key), ext_val);

		}

		__device__ __inline__ bool atomic_swap_tombstone(Key const ext_key, Val const ext_val){
	
			return storage.atomic_swap_tombstone(get_smallKey(ext_key), ext_val);

		}

		__host__ __device__ inline bool is_empty(){
			return storage.is_empty();
		}

		__host__ __device__ inline bool contains(Key ext_key){

			SmallKey tiny_version = get_smallKey(ext_key);

			return storage.contains(get_smallKey(ext_key));
		}

		__host__ __device__ inline bool contains_tombstone(){

			return storage.contains_tombstone();

		}

		__host__ __device__ inline Val get_val(Key ext_key){
			return storage.get_val(get_smallKey(ext_key));
		}

		__host__ __device__ inline void reset(){
			storage.reset();
		}

		__host__ __device__ inline static const SmallKey get_tombstone(){

			// if (threadIdx.x+blockIdx.x*blockDim.x == 2){
			// printf("Dynamic container sees tombstone as %llx\n", filled_container_type::get_tombstone());
			// }

			return filled_container_type::get_tombstone();
		}

		__device__ __inline__ bool atomic_reset(Key const ext_key){

			return storage.atomic_reset(get_smallKey(ext_key));

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

//rubber ducky strucky
//this will allow you to set the size of smallKey inside of key_val_pair
//while still allowing all the components to connect nicely.
template<template <typename, typename> class Container, typename SmallKey> struct dynamic_container {
    template<typename Key, typename Val>
    using representation = internal_dynamic_container<Container, Key, Val, SmallKey>;
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


template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator<(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){


	return A.key < B.key;

}

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator<=(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){

	return A.key <= B.key;

}

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator>=(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){

	return A.key >= B.key;

}

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator==(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){

	return A.key == B.key;

}

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator!=(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){

	return A.key != B.key;

}

template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ bool operator>(internal_dynamic_container<Container, Key, Val, SmallKey> A, internal_dynamic_container<Container, Key, Val, SmallKey> B){

	return A.key > B.key;

}


template <template <typename, typename> class Container, typename Key, typename Val, typename SmallKey>
__host__ __device__ Key operator/(internal_dynamic_container<Container, Key, Val, SmallKey> A, uint64_t other){

	return A.key / other;
}

}

}


#endif //GPU_BLOCK_