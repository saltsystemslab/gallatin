#ifndef GALLATIN_DEV_HOST_QUEUE
#define GALLATIN_DEV_HOST_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>


namespace gallatin {

namespace data_structs {



	//basic form of queue using allocator
	//on instantiation on host or device, must be plugged into allocator.
	//This allows the queue to process memory dynamically.



	//Basic ring queue for efficiently enqueueing large # of items
	//Dequeue is handled in bulk by host.
	template <typename T>
	struct device_host_queue {

		T * items;

		uint64_t nitems; 

		uint64_t enqueue_counter;


		__host__ __device__ void init(uint64_t ext_nitems){
			
			nitems = ext_nitems;
			items = (T *) global_malloc(sizeof(T)*ext_nitems);
			enqueue_counter = 0;
		}

		__device__ bool enqueue(T new_item){


			uint64_t enqueue_slot = atomicAdd((unsigned long long int *)&enqueue_counter, 1ULL);

			if (enqueue_counter >= nitems) return false;

			items[enqueue_slot] = new_item;

			__threadfence();

			return;
			
		}

		//valid to make optional type?

		__device__ uint64_t get_active_nitems(){

			if (enqueue_counter > nitems) return nitems;

			return enqueue_counter;

		}

		//copy items to new array in parallel.
		//this is needed for dumping the queue items to a buffer during output.
		__device__ void dump_to_array(T * ext_array, uint64_t my_item){

			ext_array[my_item] = items[my_item];


		}



	};

}


}


#endif //end of queue name guard