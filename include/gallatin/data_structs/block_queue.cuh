#ifndef GALLATIN_BLOCK_QUEUE
#define GALLATIN_BLOCK_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <poggers/allocators/alloc_utils.cuh>

#define DIVISION_BIT 32


namespace gallatin {

namespace data_structs {


	//storage inside of the queue.
	//this data struct is a dynamic fusion of an array-based and linked-list-based queue.
	//blocks contain multiple items and use a counter to separate them.	
	//this amortizes the cost of enqueue and dequeue as most items proceed with 1 atomic op.
	template <typename T, int num_items>
	struct block_queue_node {

		T items[num_items];
		block_queue_node<T> * next;

		//fused counters for ops - block size is a maximum of 16 billion.
		uint enqueue_counter;

		int live;

		uint dequeue_counter;

		__device__ void init(){
			//items[0] = new_item;
			next = nullptr;
			enqueue_counter = 0;
			live = 0;
			dequeue_counter = 0;
		}

		__device__ void set_next(queue_node<T> * ext_next){
			next = ext_next;
		}


		static device int get_max(){
			return num_items;
		}

		//returns the address claimed, malloc occupies the lower bits. 
		//does not do boundary checking
		__device__ uint enqueue_increment_next(){

			return (atomicAdd((unsigned int *) &enqueue_counter, 1U));

		}


		__device__ bool enqueue(T item){

			uint enqueue_pos = enqueue_increment_next();

			//this segment is empty.
			if (enqueue_pos >= num_items) return false;

			//else write!
			place_item(item, enqueue_pos);

			int num_active = signal_active();

			if (enqueue_pos != num_active){
				printf("Discrepancy: %u != %d\n", enqueue_pos, num_active);
			}

			return true;

		}

		//attempt to read item from this queue block.
		//pass in item & so that the user gets to deal with that 
		// as we don't know what the ty
		__device__ bool dequeue(T & item){

			success = grab_active();

			if (success){

				uint read_addr = get_dequeue_position();

				item = items[read_addr];


			}

			return success;
		}

		__device__ bool grab_active(){

			uint old = atomicSub((int *)&live, 1);

			if (old > 0) return true;

			atomicAdd((int *)&live, 1);

			return false;

		}


		//this may violate a precondition.
		//if so, need to use looped CAS.
		__device__ int signal_active(){

			return atomicAdd((int *) &live, 1);

		}

		//given the next node that should be in the chain and CAS
		//returns nullptr on correct swap
		//else returns new tail.
		__device__ queue_node<T> * CAS_and_return(queue_node<T> * next_node){


			return (queue_node<T> *) atomicCAS((unsigned long long int *)&next, 0ULL, (unsigned long long int )next_node);

		}

		//dequeue a segment
		//does not check if too large/too small
		//NOTE: This gives you an exclusive slot - does not guarantee
		__device__ uint get_dequeue_position(){

			return atomicAdd(((unsigned int *) &dequeue_counter, 1ULL));


		}

		//Insert items with non_atomic + threadfence?
		__device__ void place_item(T item, uint write_pos){

			items[write_pos] = item;
			__threadfence();

		}


		//use ldca to force a global read of T
		//if bigger than 64_bits, use indirection
		//no idea if this works LMAO
		__device__ T global_read(uint pos){

			return ((T *)  gallatin::utils::ldca((void *) (items + pos)))[0];

		}


	};

	//basic form of queue using allocator
	//on instantiation on host or device, must be plugged into allocator.
	//This allows the queue to process memory dynamically.



	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current

	template <typename T, int items_per_block>
	struct block_queue {

		using my_type = block_queue<T, items_per_block>;
		using node_type = block_queue_node<T, items_per_block>;

		queue_node<T> * head;
		queue_node<T> * tail;



		//instantiate a queue on device.
		//currently does not pull from the allocator, but it totally should
		static __host__ my_type * generate_on_device(){

			my_type * host_version = poggers::utils::get_host_version<my_type>();

			//host_version->my_backing_allocator = backing_allocator;

			return poggers::utils::move_to_device<my_type>(host_version);


		}

		__device__ void init(allocator * backing_allocator){
			head = nullptr;
			tail = nullptr;
		}

		__device__ void enqueue(T new_item){

			my_head = head;

			while (true){


				if (my_head == nullptr){

					//create new_node
					node_type * next_node = global_malloc(sizeof(next_node));

					next_node->init();

					__threadfence();

					if (atomicCAS((unsigned long long int *)&head, 0ULL, (unsigned long long int)next_node) == 0ULL){

						atomicCAS((unsigned long long int *)&tail, 0ULL, (unsigned long long int) next_node);

						continue;
					}

					global_free(next_node);



				}

			}

			

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