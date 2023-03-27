#ifndef SIZE_IN_NUM_SLOTS 
#define SIZE_IN_NUM_SLOTS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <cstdarg>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace sizing {






//probing schemes map keys to buckets/slots in some predefined pattern

//template<typename X, template <typename> class SomeSmall> 
template <std::size_t Num_Tables>
//template <typename Hasher, std::size_t Max_probes>
struct __attribute__ ((__packed__)) size_in_num_slots {


	//tag bits change based on the #of bytes allocated per block
private:


	uint64_t slots [Num_Tables];

	uint64_t i;


public:


	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	__host__  size_in_num_slots(uint64_t start, ...){

		i = 0;
		std::va_list args;
		va_start(args, start);
		for (uint64_t j=1; j < Num_Tables; j++){
			slots[j] = va_arg(args, uint64_t);
		}
		va_end(args);

		slots[0] = start;

	}


	__host__ uint64_t next(){

		i+=1;
		return i <= Num_Tables ? slots[i-1] : end();
	}

	__host__ void reset(){
		i = 0;
	}

	__host__ uint64_t end(){
		return ~uint64_t(0);
	}

	__host__ uint64_t total(){
		uint64_t total = 0;

		for (uint64_t j=0; j < Num_Tables; j++){
			total+=slots[j];
		}
		return total;
	}


	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_