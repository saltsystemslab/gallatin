#ifndef REPORTER
#define REPORTER


#include <cuda.h>
#include <cuda_runtime_api.h>


#include "stdio.h"
#include "assert.h"


//A series of inclusions for building a poggers hash table


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//CMS reporter
//This object traces over all allocated structure to get a summary of memory in use.
//along with some tools to nicely spit out the results
//This is generated when you call shibboleth->report();




//a pointer list managing a set section o fdevice memory

// 	const float log_of_size = std::log2()

// }

namespace poggers {


namespace allocators { 



struct reporter {


	using my_type = reporter;


	// uint64_t stack_bytes_malloced;
	// uint64_t stack_bytes_free;
	// uint64_t dead_list_malloced;
	// uint64_t dead_list_free;
	// uint64_t heap_bytes_total;
	// uint64_t heap_bytes_free;
	uint64_t recordings[10];



	__device__ void init(){


		for (int i=0; i< 10; i++){
			recordings[i] = 0;
		}
	}


	__device__ void modify_recording(int modification, int selection){

		while (true){
			uint64_t old_value = recordings[selection];

			uint64_t new_value = old_value+modification;

			if (atomicCAS((unsigned long long int  *)&recordings+selection, old_value, new_value) == old_value){
				return;
			}
		}
	}

	__device__ void modify_stack_bytes_malloced(int modification){

		modify_recording(modification, 0);

	}
	__device__ void modify_stack_bytes_free(int modification){

		modify_recording(modification, 1);

	}

	__device__ void modify_dead_bytes_malloced(int modification){

		modify_recording(modification, 2);

	}
	__device__ void modify_dead_bytes_free(int modification){

		modify_recording(modification, 3);

	}

	__device__ void modify_heap_bytes_total(uint64_t modification){

		const int cutoff = 1000000000;

		while (modification > cutoff){

			modify_recording(cutoff, 4);
			modification -= cutoff;

		}

		modify_recording( (int) modification, 4);

	}

	__device__ void modify_heap_bytes_free(uint64_t modification){

		const int cutoff = 1000000000;

		while (modification > cutoff){

			modify_recording(cutoff, 5);
			modification -= cutoff;

		}

		modify_recording( (int) modification, 5);

	}

	__device__ void modify_fragmentation_count(uint64_t modification){

		const int cutoff = 1000000000;

		while (modification > cutoff){

			modify_recording(cutoff, 8);
			modification -= cutoff;

		}

		modify_recording( (int) modification, 8);

	}

	__device__ void modify_num_stacks(int modification){

		modify_recording(modification, 6);
	}

	__device__ void modify_dead_stacks(int modification){
		modify_recording(modification, 7);
	}
 
 	__device__ uint64_t get_stack_bytes_malloced(){

		return recordings[0];

	}
	__device__ uint64_t get_stack_bytes_free(){

		return recordings[1];

	}

	 __device__ uint64_t get_dead_bytes_malloced(){

		return recordings[2];

	}
	__device__ uint64_t get_dead_bytes_free(){

		return recordings[3];

	}

	__device__ uint64_t get_heap_bytes_total(){

		return recordings[4];

	}

	__device__ uint64_t get_heap_bytes_free(){

		return recordings[5];

	}

	__device__ uint64_t get_total_stacks(){
		return recordings[6];
	}

	__device__ uint64_t get_dead_stacks(){
		return recordings[7];
	}

	__device__ uint64_t get_fragmentation(){
		return recordings[8];
	}


};


}

}


#endif //GPU_BLOCK_