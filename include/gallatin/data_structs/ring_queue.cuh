#ifndef GALLATIN_RING_QUEUE
#define GALLATIN_RING_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

namespace gallatin {

namespace data_structs {


	//basic form of queue using allocator
	//on instantiation on host or device, must be plugged into allocator.
	//This allows the queue to process memory dynamically.



	//Pipeline

	//insert op
	

	template <typename T>
	__global__ void init_ring_kernel(T * buffer, T default_value, uint64_t num_slots){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= num_slots) return;

		buffer[tid] = default_value;

	}

	template <typename T, T default_value>
	struct ring_queue {

		using my_type = ring_queue<T, default_value>;


		uint64_t num_slots;
		T * buffer;


		int active_count;

		uint64_t enqueue_counter;
		uint64_t dequeue_counter;

		//instantiate a queue on device.
		//currently does not pull from the allocator, but it totally should
		static __host__ my_type * generate_on_device(uint64_t ext_num_slots){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->num_slots = ext_num_slots;


			T * ext_buffer;

			cudaMalloc((void **)&ext_buffer, sizeof(T)*ext_num_slots);

			init_ring_kernel<T><<<(ext_num_slots-1)/256+1,256>>>(ext_buffer, default_value, ext_num_slots);

			host_version->buffer = ext_buffer;
			host_version->enqueue_counter = 0;
			host_version->dequeue_counter = 0;
			host_version->active_count = 0;

			cudaDeviceSynchronize();

			return gallatin::utils::move_to_device<my_type>(host_version);


		}


		static __host__ void free_on_device(my_type * dev_queue){

			my_type * host_version = gallatin::utils::move_to_host(dev_queue);

			cudaFree(host_version->buffer);
			cudaFreeHost(host_version);

		}

		__device__ bool enqueue(T new_item){


			while (true){


			
				int slot_active_count = atomicAdd(&active_count, 1);

				//this should be ok. double check this later.
				//if (slot_active_count < 0)

				//cycle to prevent overcount.
				if (slot_active_count < 0){
					atomicSub(&active_count, 1);
					continue;
				}


				if (slot_active_count >= num_slots){

					atomicSub(&active_count, 1);

					return false;
				}

				//is slot_active_count + live_dequeue same as atomic?

				uint64_t enqueue_slot = atomicAdd((unsigned long long int *)&enqueue_counter,1ULL);

				//needs to loop.
				while(typed_atomic_CAS(&buffer[enqueue_slot], default_value, new_item) != default_value);
				// 	//throw error
				// 	asm volatile("trap;"); 
				// 	return false;
				// }

				return true;

			}

		}

		//valid to make optional type?

		__device__ bool dequeue(T & return_val){

			//do these reads need to be atomic? 
			//I don't think so as these values don't change.
			//as queue doesn't change ABA not possible.
			while (true){


				int slot_active_count = atomicSub(&active_count, 1);

				if (slot_active_count <= 0){

					atomicAdd(&active_count, 1);
					return false;
				}


				if (slot_active_count > num_slots){
					//cycle

					atomicAdd(&active_count, 1);

					continue;
				}

				//slot is valid!

				uint64_t dequeue_slot = atomicAdd((unsigned long long int *) &dequeue_counter, 1ULL);
				

				return_val = typed_atomic_exchange(&buffer[dequeue_slot], default_value);

				while (return_val == default_value){
					return_val = typed_atomic_exchange(&buffer[dequeue_slot], default_value);
				}

				return true;

			}


		}




	};


	// //this queue uses a circular ring of queues to encode items
	// // - this reduces memory pressure on individual queues
	// template<typename T, int width, typename allocator>
	// struct circular_CAS_queue {

	// 	using my_type = circular_CAS_queue<T, width, allocator>

	// 	uint64_t malloc_counter;
	// 	uint64_t free_counter;
	// 	queue queue_list[width];

	// 	__host__ __device__ void init(allocator * backing_allocator){


	// 		malloc_counter = 0;
	// 		free_counter = 0;

	// 		for (int i = 0; i < width; i++){
	// 			queue_list[i].init(allocator);
	// 		}

	// 	}


	// 	//generate a live version of the queue
	// 	__host__ my_type * generate_on_device(allocator * backing_allocator){

	// 		my_type * host_version = gallatin::utils::get_host_version<my_type>();

	// 		host_version->init(backing_allocator);

	// 		return gallatin::utils::move_to_device<my_type>(host_version);

	// 	}

	// 	__device__ void enqueue(T new_item){

	// 		uint64_t my_count = atomicAdd((unsigned long long int *)&malloc_counter, 1ULL) % width;

	// 		queue_list[my_count].enqueue(new_item);


	// 	}


	// 	__device__ bool dequeue(T & return_val){

	// 		uint64_t my_count = atomicAdd((unsigned long long int *)&free_counter, 1ULL) % width;

	// 		return queue_list[my_count].dequeue(return_val);

	// 	}



	// };


}


}


#endif //end of queue name guard