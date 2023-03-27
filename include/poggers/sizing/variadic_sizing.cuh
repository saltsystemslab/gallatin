#ifndef VARIADIC_SIZE_IN_NUM_SLOTS 
#define VARIADIC_SIZE_IN_NUM_SLOTS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <cstdarg>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


/*****************************
 * 
 * 
 * IN PROGRESS - There seems to be a problem with resetting variadic sized tables
 * If possible, stick with default sizing until this is patched.
 * 
 */


namespace poggers {

namespace sizing {






//probing schemes map keys to buckets/slots in some predefined pattern

//template<typename X, template <typename> class SomeSmall> 
//template <typename Hasher, std::size_t Max_probes>
struct __attribute__ ((__packed__)) variadic_size {


	//tag bits change based on the #of bytes allocated per block
private:


	uint64_t * slots;

	int num_slots;

	int i;


public:


	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage


	template <typename T, typename... Targs>
	__host__ void recursive_init(int i, T first_item, Targs...rest){

		

		slots[i] = first_item;

		if constexpr (sizeof...(rest) > 0) recursive_init(i+1, rest...);

	}

	template<typename... T>
	__host__  variadic_size(T...items){


		const std::size_t n = sizeof...(T);

		slots = (uint64_t *) malloc(sizeof(uint64_t)*n);

		num_slots = n;



		recursive_init(0, items...);

		// i = 0;
		// std::va_list args;
		// va_start(args, start);
		// for (int i=1; i < num_slots; i++){
		// 	slots[i] = va_arg(args, uint64_t);
		// }
		// va_end(args);

		// slots[0] = start;

		for (int i =0; i < num_slots; i++){
			printf("%llu\n", slots[i]);
		}


	}


	__host__ uint64_t next(){

		i+=1;
		return i <= num_slots ? slots[i-1] : end();
	}


	__host__ uint64_t end(){
		return ~uint64_t(0);
	}

	__host__ void reset(){
		i = 0;
	}

	__host__ uint64_t total(){
		uint64_t total = 0;

		for (int i=0; i < num_slots; i++){
			total+=slots[i];
		}
		return total;
	}

	~variadic_size(){

		free(slots);
	}

	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_