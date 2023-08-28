#ifndef GALLATIN_QUEUE
#define GALLATIN_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <poggers/allocators/alloc_utils.cuh>


namespace gallatin {

namespace data_structs {


	//storage inside of the queue.
	//sizeof(queue_node) is requested from the allocator, and then emplaced using atomicCAS
	template <typename Key, typename Val, int bucket_size, int Partition_size>
	struct hash_segment {

		using my_type = hash_segment<Key, Val>;


		uint64_t num_slots;
		uint64_t seed;
		my_type * next;


		//given a section of memory, initialize the hash segment.
		//this zeros out the memory.
		__device__ static my_type * init(void * memory, uint64_t seed, uint64_t num_bytes){

			my_type * segment = (my_type *) memory;

			segment->num_slots = (num_bytes - 32)/segment->get_stride();

			segment->seed = seed;

			return segment;



		}

		static __host__ __device__ const uint get_stride(){

			return poggers::utils::rounded_size<Key, Val>::size;




		}

		__device__ Key * get_key_addr(uint64_t slot){

			const uint stride = my_type::get_stride();

			Key * key_addr = (Key *) (((char * ) this) + 32 + stride*slot);

			return key_addr;

		}

		__device__ void insert(Key key, Val val){

			poggers::hashers::murmurHasher hash_func;
			hash_func.init(seed);

			uint64_t hash = hash_func.hash(key);

		}

		__device__ bool query(Key key, Val & val){
			
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

			my_type * host_version = poggers::utils::get_host_version<my_type>();

			host_version->my_backing_allocator = backing_allocator;

			return poggers::utils::move_to_device<my_type>(host_version);


		}

		__device__ void init(allocator * backing_allocator){
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


}


}


#endif //end of queue name guard