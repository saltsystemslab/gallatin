#ifndef MURMUR_HASHER 
#define MURMUR_HASHER


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


namespace poggers {

namespace hashers {


__host__ __device__ uint64_t MurmurHash64A ( const void * key, int len, uint64_t seed )
{
	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const uint64_t * data = (const uint64_t *)key;
	const uint64_t * end = data + (len/8);

	while(data != end)
	{
		uint64_t k = *data++;

		k *= m; 
		k ^= k >> r; 
		k *= m; 

		h ^= k;
		h *= m; 
	}

	const unsigned char * data2 = (const unsigned char*)data;

	switch(len & 7)
	{
		case 7: h ^= (uint64_t)data2[6] << 48;
		case 6: h ^= (uint64_t)data2[5] << 40;
		case 5: h ^= (uint64_t)data2[4] << 32;
		case 4: h ^= (uint64_t)data2[3] << 24;
		case 3: h ^= (uint64_t)data2[2] << 16;
		case 2: h ^= (uint64_t)data2[1] << 8;
		case 1: h ^= (uint64_t)data2[0];
						h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}






//hashers should always hash an arbitrary type to uint64_t
template <typename Key, std::size_t Partition_size = 1>
struct __attribute__ ((__packed__)) murmurHasher {


	//tag bits change based on the #of bytes allocated per block
private:
	uint64_t seed;

public:

	using key_type = Key;
	//using internal_paritition_size = Partition_size;
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	__host__ __device__ void init(uint64_t ext_seed){	

		seed = ext_seed;
	}


	__host__ __device__ uint64_t hash(Key key_to_hash){

		Key copy_key = key_to_hash;
		
		return MurmurHash64A(&copy_key, sizeof(Key), seed);


	}

	//all participate
	__device__ uint64_t hash(Key key_to_hash, cg::thread_block_tile<Partition_size> group){

		Key copy_key = key_to_hash;

		return MurmurHash64A(&copy_key, sizeof(Key), seed);

	}


	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_