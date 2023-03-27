#ifndef GLOBAL_HEAP
#define GLOBAL_HEAP


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>



// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace allocators { 


//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
template <size_t Reservation_Size, size_t Max_Size>
struct  superblock {

	public:

		void * head;



		__host__ __device__ superblock(){};


		__device__ partition_void_pointer(void * allocated_space){

			//allocate

		}


		__device__ void * request_node(uint64_t num_items){

		}



		

};



}

}


#endif //GPU_BLOCK_