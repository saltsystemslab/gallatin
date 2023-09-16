#ifndef GALLATIN_QUEUE
#define GALLATIN_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/alloc_utils.cuh>


namespace gallatin {

namespace data_structs {


	//storage inside of the queue.
	//sizeof(queue_node) is requested from the allocator, and then emplaced using atomicCAS
	template <typename T>
	struct queue_node {

		T item;
		queue_node<T> * next;

		__device__ void init(T new_item){
			item = new_item;
			next = nullptr;
		}

		__device__ void set_next(queue_node<T> * ext_next){
			next = ext_next;
		}

		//given the next node that should be in the chain and CAS
		//returns nullptr on correct swap
		//else returns new tail.
		__device__ queue_node<T> * CAS_and_return(queue_node<T> * next_node){


			return (queue_node<T> *) atomicCAS((unsigned long long int *)&next, 0ULL, (unsigned long long int )next_node);

		}


	};

	//basic form of queue using allocator
	//on instantiation on host or device, must be plugged into allocator.
	//This allows the queue to process memory dynamically.



	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current

	template <typename T, typename allocator>
	struct queue {

		using my_type = queue<T, allocator>;

		queue_node<T> * head;
		queue_node<T> * tail;

		allocator * my_backing_allocator;


		//instantiate a queue on device.
		//currently does not pull from the allocator, but it totally should
		static __host__ my_type * generate_on_device(allocator * backing_allocator){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->init(backing_allocator);

			return gallatin::utils::move_to_device<my_type>(host_version);


		}

		__host__ __device__ void init(allocator * backing_allocator){
			my_backing_allocator = backing_allocator;
			head = nullptr;
			tail = nullptr;
		}

		__device__ void enqueue(T new_item){

			queue_node<T> * new_node = (queue_node<T> *) my_backing_allocator->malloc(sizeof(queue_node<T>));

			if (new_node == nullptr){
				printf("Failed to enqueue - could not acquire node\n");
				return;
			}

			new_node->init(new_item);

			__threadfence();

			//now that node is init + visible, add to system.
			if (tail == nullptr){
				if (atomicCAS((unsigned long long int *)&tail, 0ULL, (unsigned long long int)new_node) == 0ULL){

					atomicExch((unsigned long long int *)&head, (unsigned long long int)new_node);

					__threadfence();

					return;

				}
			}


			queue_node<T> * current_node = tail->CAS_and_return(new_node);


			while (current_node != nullptr){

				current_node = current_node->CAS_and_return(new_node);

			}

			//swap current node into tail so that future threads observe closer tail.
			//this guarantees progress from start of this tail but may be marginal if schedule is weird
			//test this.
			//would it be faster to make this an atomicCAS? guarantees that tail is monotonic.
			//atomicExch((unsigned long long int *)&tail, (unsigned long long int)current_node);


			atomicCAS((unsigned long long int *)&tail, (unsigned long long int )current_node, (unsigned long long int) new_node);
			
			__threadfence();

		}

		//valid to make optional type?

		__device__ bool dequeue(T & return_val){

			//do these reads need to be atomic? 
			//I don't think so as these values don't change.
			//as queue doesn't change ABA not possible.

			__threadfence();

			queue_node<T> * my_head = head;

			if (head == nullptr){
				return false;
			}

			queue_node<T> * head_next = my_head->next;

			queue_node<T> * swap = (queue_node<T> *) atomicCAS((unsigned long long int *)&head, (unsigned long long int)my_head, (unsigned long long int)head_next);
			
			while (swap != my_head){
				__threadfence();

				my_head = swap;

				if (head == nullptr){
					return false;
				}


				head_next = my_head->next;
				swap = (queue_node<T> *) atomicCAS((unsigned long long int *)&head, (unsigned long long int)my_head, (unsigned long long int)head_next);
			


			}

			//at the end my_head is valid

			__threadfence();

			return_val = my_head->item;

			//would be nice to have a hazard pointer here.

			my_backing_allocator->free(my_head);

			return true;

		}




	};


	//this queue uses a circular ring of queues to encode items
	// - this reduces memory pressure on individual queues
	template<typename T, int width, typename allocator>
	struct circular_CAS_queue {

		using my_type = circular_CAS_queue<T, width, allocator>;

		uint64_t malloc_counter;
		uint64_t free_counter;
		queue queue_list[width]; 

		__host__ __device__ void init(allocator * backing_allocator){


			malloc_counter = 0;
			free_counter = 0;

			for (int i = 0; i < width; i++){
				queue_list[i].init(allocator);
			}

		}


		//generate a live version of the queue
		__host__ my_type * generate_on_device(allocator * backing_allocator){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->init(backing_allocator);

			return gallatin::utils::move_to_device<my_type>(host_version);

		}

		__device__ void enqueue(T new_item){

			uint64_t my_count = atomicAdd((unsigned long long int *)&malloc_counter, 1ULL) % width;

			queue_list[my_count].enqueue(new_item);


		}


		__device__ bool dequeue(T & return_val){

			uint64_t my_count = atomicAdd((unsigned long long int *)&free_counter, 1ULL) % width;

			return queue_list[my_count].dequeue(return_val);

		}



	};


}


}


#endif //end of queue name guard