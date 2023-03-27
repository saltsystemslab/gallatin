#ifndef LINEAR_PROBING 
#define LINEAR_PROBING


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace probing_schemes {






//probing schemes map keys to buckets/slots in some predefined pattern

//template<typename X, template <typename> class SomeSmall> 
template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_probes>
//template <typename Hasher, std::size_t Max_probes>
struct __attribute__ ((__packed__)) linearProber {


	//tag bits change based on the #of bytes allocated per block
private:
	Hasher<Key, Partition_Size> my_hasher;

	int depth;

	uint64_t hash;

	int i;



public:

	//typedef key_type Hasher::key_type;
	using key_type = typename Hasher<Key, Partition_Size>::key_type;
 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	__host__ __device__ linearProber(uint64_t seed){
		my_hasher.init(seed);
	}


	__host__ __device__ uint64_t begin(key_type ext_key){

		hash = my_hasher.hash(ext_key);
		i = 0;

		return hash;
	}

	__host__ __device__ uint64_t next(key_type tag){

		i+=1;

		return i < Max_probes ? hash+i : end();

	}

	__host__ __device__ uint64_t end(){

		return ~uint64_t(0);
	}


	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_