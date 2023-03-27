#ifndef XOR_P2_HASHING_PROBING 
#define XOR_P2_HASHING_PROBING


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace probing_schemes {



//hash scheme for dependence between buckets


//probing schemes map keys to buckets/slots in some predefined pattern
template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_probes>
struct __attribute__ ((__packed__)) XORPowerOfTwoHasher {


	//tag bits change based on the #of bytes allocated per block
private:
	Hasher<Key, Partition_Size> my_first_hasher;

	int depth;

	uint64_t hash1;

	int i;



public:



	//typedef key_type Hasher::key_type;
	using key_type = Key;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	__host__ __device__ XORPowerOfTwoHasher(uint64_t seed){
		my_first_hasher.init(seed);
	}


	__host__ __device__ uint64_t begin(key_type ext_key){

		hash1 = my_first_hasher.hash(ext_key);

		i = 0;

		return hash1;
	}

	__host__ __device__ uint64_t next(key_type tag){

		i+=1;

		return i < Max_probes ? hash1 ^ (tag * 0x5bd1e995) : end();

	}

	__host__ __device__ uint64_t end(){

		return ~uint64_t(0);
	}


	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_