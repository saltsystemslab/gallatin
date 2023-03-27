#ifndef TEMPLATED_FUNCS 
#define TEMPLATED_FUNCS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace experimental {



//The insert schemes are in charge of device side inserts
//they handle memory allocations for the host-side table
// and most of the logic involved in inserting/querying


//Power of N Hashing
//Given a probing scheme with depth N, this insert strategy
//queries the fill of all N buckets

//TODO = separate bucket size from cuda size - use static assert to enforce correctness
// bucket_size/NUM_BUCKETS

//TODO - get godbolt to compile a cmake project? that would be nice.
//template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_Probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_Probes>
struct __attribute__ ((__packed__)) example_counter {

	

public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	//using my_type = example_counter<uint64_t>;


	//using partition_size = Hasher1::Partition_Size;
	uint counter;
 
	
	//typedef key_uint64_t_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ example_counter(): counter(0) {};


	//only allowed to be defined on CPU
	__host__ __device__ example_counter(uint64_t ext_counter): counter(ext_counter){}


	__device__ void ext_atomicInc(uint toAdd){

		printf("To add: %llu\n", toAdd);

		atomicAdd((unsigned int *) &counter, (unsigned int) toAdd);
	}

	__device__ void ext_atomicDec(uint toDec){

		atomicSub((unsigned int *) &counter, (unsigned int) toDec);
	}

	__device__ void dud(){

		printf("I'm in danger!\n");
		return;
	}



};


// template <void(example_counter::*func)(uint)>
// void Call(example_counter * test, uint val){
// 	(test->*func)(val);
// }

template <void(example_counter::*func)(uint)>
__device__ void Int_call(example_counter * test, uint val){
	(test->*func)(val);
}

template <void(example_counter::*func)()>
__device__ void Call(example_counter * test){
	(test->*func)();
}

//experimental
}

//poggers
}


#endif //GPU_BLOCK_