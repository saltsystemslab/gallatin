#ifndef GROUPED_KEY_VAL_PAIR_H 
#define GROUPED_KEY_VAL_PAIR_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/representations/representation_helpers.cuh>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace representations { 


//Template metaprogramming to determine minimum container

// template<typename T>
// struct return_
// {
//     typedef T type;
// };

// //given bytes used generate the main
// template<uint64_t N>
// struct bytetype : return_<uint64_t> {};

// template<>
// struct bytetype<4> : return_<uint32_t> {};

// template<>
// struct bytetype<3> : return_<uint32_t> {};

// template<>
// struct bytetype<2> : return_<uint16_t> {};

// template<>
// struct bytetype<1> : return_<uint8_t> {};


//recursive structure for p02 padding



//Storage keeps items (Val, Key) in memory
//Key retreived via OR with lower bits
template<typename Key, typename Val>
struct bit_internal_key_val_storage : poggers::helpers::bytetype<sizeof(Key)+sizeof(Val)>{};


template<typename Key, typename Val, typename Storage>
__host__ __device__ Key retrieve_key_from_storage(Storage my_storage){

	//split 
	return (my_storage & (((Storage{1}) << sizeof(Key)*8) -1));

	// Storage mask = ((((Storage) 0) << (sizeof(Val)*8)) -1) << sizeof(Key);

	// my_val = my_storage & mask;

}

template<typename Key, typename Val, typename Storage>
__host__ __device__ Val retrieve_val_from_storage(Storage my_storage){

	//split 
	//my_key = my_storage & ((((Storage) 0) << sizeof(Key)*8) -1);

	Storage lower_bits = (((Storage{1}) << (sizeof(Val)*8)) -1);

	Storage mask = lower_bits << sizeof(Key)*8;

	// printf("Lower bits %lx, Mask is %lx\n", lower_bits, mask);

	// printf("Storage is %lx, Mask is %lx\n", my_storage, mask);

	Val my_val = (my_storage & mask) >> sizeof(Key)*8;

	return my_val;

}


template <typename Key, typename Val, typename Storage>
__host__ __device__ Storage join_in_storage(Key my_key, Val my_val){

	Storage empty_storage = Storage{0};

	//printf("Storing %lx, %u, %u\n", empty_storage, my_key, my_val);

	empty_storage |= my_val << 8*sizeof(Key);

	empty_storage |= my_key;

	return empty_storage;

}



template <typename Key, typename Val>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  grouped_key_val_pair {

	private:
		

	public:

		using storage_type = typename bit_internal_key_val_storage<Key,Val>::type;

		storage_type my_storage;

		__host__ __device__ static inline const Key get_empty(){ return Key{0}; }

		__host__ __device__ static inline const Val get_empty_val(){ return Val{0}; }


		__host__ __device__ grouped_key_val_pair(){}

		//constructor
		__host__ __device__ grouped_key_val_pair (Key const & key, Val const & val)
		: my_storage(join_in_storage<Key, Val, storage_type>(key, val)){}

		__device__ __inline__ bool atomic_swap(Key const ext_key, Val const ext_val){

			storage_type ext_storage = join_in_storage<Key, Val, storage_type>(ext_key, ext_val);

			if (poggers::helpers::typed_atomic_write(&my_storage, join_in_storage<Key, Val, storage_type>(get_empty(), get_empty_val()), ext_storage)){

				//val = ext_val;

				return true;

			}

			return false;

		}

		__device__ __inline__ bool atomic_swap_tombstone(Key const ext_key, Val const ext_val){

			storage_type ext_storage = join_in_storage<Key, Val, storage_type>(ext_key, ext_val);

			if (poggers::helpers::typed_atomic_write(&my_storage, join_in_storage<Key, Val, storage_type>(get_tombstone(), get_empty_val()), ext_storage)){

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


		__host__ __device__ inline bool is_empty(){

			
			Key key = retrieve_key_from_storage<Key, Val, storage_type>(my_storage);

			return (key == get_empty());
		}

		__host__ __device__ inline static const Key get_tombstone(){
			return get_empty()-1;
		}

		__host__ __device__ inline bool contains(Key ext_key){

			Key key = retrieve_key_from_storage<Key, Val, storage_type>(my_storage);

			return (key == ext_key);
		}


		__device__ inline bool contains_tombstone(){
			return contains(get_tombstone());
		}

		__host__ __device__ inline Val get_val(Key ext_key){

			Val val = retrieve_val_from_storage<Key, Val, storage_type>(my_storage);

			return val;
		}

		__host__ __device__ inline void reset(){

			storage_type reset_storage = join_in_storage<Key, Val, storage_type>(get_empty(), get_empty_val());

			//storage_type ext_storage = join_in_storage<Key, Val, storage_type>()

			poggers::helpers::typed_atomic_write(&my_storage, my_storage, reset_storage);



		}

		static __device__ inline Key tag(Key ext_key){

			storage_type in_storage = join_in_storage<Key, Val, storage_type>(ext_key, get_empty_val());

			return retrieve_key_from_storage<Key, Val, storage_type>(in_storage);


		}

		__device__ __inline__ bool atomic_reset(Key const ext_key){

			Val my_val = get_val(ext_key);

			storage_type ext_storage = join_in_storage<Key, Val, storage_type>(ext_key, my_val);

			storage_type tombstone_storage = join_in_storage<Key, Val, storage_type>(get_tombstone(), get_empty_val());


			if (poggers::helpers::typed_atomic_write(&my_storage, ext_storage, tombstone_storage)){
				return true;
			}

			return false;

		}
		

};


// template <typename Key, typename Val>
// __device__ void pack_into_pair(grouped_key_val_pair<Key, Val> & pair, Key & new_key, Val & new_val ){

// 	pair.set_key(new_key);
// 	pair.set_val(new_val);


// }

// template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
// struct grouped_key_val_pair{


// 	//tag bits change based on the #of bytes allocated per block

// 	storage_pair<Key, Val> internal_storage;

// 	//grouped_key_val_pair (Key const)

// 	//you only use this when you have vals
// 	grouped_key_val_pair(Key const & key, Val const & val): storage_pair<Key, Val>(key, val){}

// 	grouped_key_val_pair(Key const & key): storage_pair<Key, Val>(key){}

// 	grouped_key_val_pair(){}

// 	__host__ __device__ Key& key{

// 		return internal_storage.key;
// 	}

// 	__host__ __device__ Val& get_val(){

// 		return internal_storage.get_val();
// 	}



// };

// template <typename Key>
// struct grouped_key_val_pair<Key, void>{



// };


template <typename Key, typename Val>
__host__ __device__ bool operator<(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){


	return A.key < B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator<=(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){

	return A.key <= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>=(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){

	return A.key >= B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator==(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){

	return A.key == B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator!=(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){

	return A.key != B.key;

}

template <typename Key, typename Val>
__host__ __device__ bool operator>(grouped_key_val_pair<Key, Val> A, grouped_key_val_pair<Key, Val> B){

	return A.key > B.key;

}


template <typename Key, typename Val>
__host__ __device__ Key operator/(grouped_key_val_pair<Key, Val >A, uint64_t other){

	return A.key / other;
}

}

}


#endif //GPU_BLOCK_