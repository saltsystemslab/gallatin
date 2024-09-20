#ifndef GALLATIN_BLOCK_QUEUE
#define GALLATIN_BLOCK_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

#define DIVISION_BIT 32


//make sure to use CAS for ensuring linearizable output.

//insertion - atomicAdd to set original slot
//atomicExch to actually set
//then need to signal availability
//then signal availability - increment turn parameter
//other threads can only enter when their count < turn_count
//read both with one atomic.

//could be improved if num_items < 32 by using a bitarray.

namespace gallatin {

namespace data_structs {


	//storage inside of the queue.
	//this data struct is a dynamic fusion of an array-based and linked-list-based queue.
	//blocks contain multiple items and use a counter to separate them.	
	//this amortizes the cost of enqueue and dequeue as most items proceed with 1 atomic op.

	//this could be
	template <typename T, int num_items>
	struct block_queue_node {

		using my_type = block_queue_node<T,num_items>;

		uint enqueue_counter;
		uint dequeue_counter;

		my_type * next;

		//upper dequeue counter processes request - returns yes if valid
		//this guarantees that every slot is unique
		uint ordered_dequeue_counter;
		uint want_enqueue_counter;

		T items[num_items];
		

		//fused counters for ops - block size is a maximum of 8 billion.
		
	

		__device__ void init(){
			//items[0] = new_item;
			//next = nullptr;

			atomicExch((unsigned long long int *)&next, 0ULL);
			atomicExch((unsigned int *)&want_enqueue_counter, 0U);
			atomicExch((unsigned int *)&enqueue_counter, 0U);
			atomicExch((unsigned int *)&dequeue_counter, 0U);
			atomicExch((unsigned int *)&ordered_dequeue_counter, 0U);
		}

		//attempt to set the address of the next node
		//return true if true.
		__device__ bool set_next(my_type * ext_next){


			return (atomicCAS((unsigned long long int *)&next, 0ULL, (unsigned long long int)ext_next) == 0ULL);
		}


		static __device__ int get_max(){
			return num_items;
		}

		//returns the address claimed, malloc occupies the lower bits. 
		//does not do boundary checking
		__device__ uint enqueue_increment_next(){

			return (atomicAdd((unsigned int *) &want_enqueue_counter, 1U));

		}


		__device__ bool enqueue(T item){

			uint enqueue_pos = enqueue_increment_next();

			//this segment is empty.
			if (enqueue_pos >= num_items) return false;

			//else write!
			place_item(item, enqueue_pos);

			signal_active(enqueue_pos);

			// if (enqueue_pos != num_active){
			// 	printf("Discrepancy: %u != %d\n", enqueue_pos, num_active);
			// }

			return true;

		}

		//attempt to read item from this queue block.
		//pass in item & so that the user gets to deal with that 
		// as we don't know what the ty
		__device__ bool dequeue(T & item){

			bool success = grab_active();

			if (success){

				uint read_addr = get_dequeue_position();

				item = (T) typed_atomic_exchange(&items[read_addr], (T)0);


			}

			return success;
		}

		__device__ bool grab_active(){

			uint64_t old = atomicAdd((unsigned long long int *)&dequeue_counter, 1ULL);

			//split

			uint64_t n_enqueued = old >> 32;

			uint64_t my_count = old & BITMASK(32);

			if (my_count > num_items) return false;

			if (n_enqueued > my_count) return true;


			atomicSub((unsigned int *)&dequeue_counter, 1U);

			return false;

		}


		//use atomicCAS to only allow correct orderings to proceed.
		__device__ void signal_active(uint previous){


			uint old = atomicCAS((unsigned int *) &enqueue_counter, (unsigned int) previous, (unsigned int)previous+1);

			while(old != previous){

				//printf("Stalling: %u != %u\n", old, previous);

				old = atomicCAS((unsigned int *) &enqueue_counter, (unsigned int) previous, (unsigned int)previous+1);


			}

			//printf("Success! %u == %u\n", old, previous);

		}

		//given the next node that should be in the chain and CAS
		//returns nullptr on correct swap
		//else returns new tail.
		__device__ my_type * CAS_and_return(my_type * next_node){


			return (my_type *) atomicCAS((unsigned long long int *)&next, 0ULL, (unsigned long long int )next_node);

		}

		//dequeue a segment
		//does not check if too large/too small
		//NOTE: This gives you an exclusive slot - does not guarantee
		__device__ uint get_dequeue_position(){

			return atomicAdd((unsigned int *) &ordered_dequeue_counter, 1U);


		}

		//Insert items with non_atomic + threadfence?
		__device__ void place_item(T item, uint write_pos){

			gallatin::utils::typed_atomic_exchange(&items[write_pos], item);
			//items[write_pos] = item;
			__threadfence();

		}


		//use ldca to force a global read of T
		//if bigger than 64_bits, use indirection
		//no idea if this works LMAO
		// __device__ T global_read(uint pos){

		// 	return ((T *)  gallatin::utils::ldca((void *) (items + pos)))[0];

		// }


	};

	//basic form of queue using allocator
	//on instantiation on host or device, must be plugged into allocator.
	//This allows the queue to process memory dynamically.



	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current

	template <typename queue>
	__global__ void queue_init_dev_version(queue * my_queue){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid != 0 )return;

		my_queue->init();

	}


	template <typename T, int items_per_block>
	struct block_queue {

		using my_type = block_queue<T, items_per_block>;
		using node_type = block_queue_node<T, items_per_block>;

		node_type * head;
		node_type * tail;



		//instantiate a queue on device.
		//currently does not pull from the allocator, but it totally should
		static __host__ my_type * generate_on_device(){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			//host_version->my_backing_allocator = backing_allocator;

			my_type * dev_version =  gallatin::utils::move_to_device<my_type>(host_version);

			queue_init_dev_version<my_type><<<1,1>>>(dev_version);

			cudaDeviceSynchronize();

			return dev_version;


		}

		//can simplify the logic a tonnnn if we allow for there to always be
		// at least one node, so head/tail never go to nullptr.
		__device__ void init(){


			node_type * head_node = (node_type *) gallatin::allocators::global_malloc(sizeof(node_type));

			head_node->init();
			head = head_node;
			tail = head_node;
		}


		//doesn't matter for the 
		__device__ void try_swap_head(node_type * old_head, node_type * my_head){


			atomicCAS((unsigned long long int *)&head, (unsigned long long int)old_head, (unsigned long long int)my_head);

		}

		__device__ node_type * add_to_tail(node_type ** current_tail, node_type *next_tail){


			return (node_type *) atomicCAS((unsigned long long int *)current_tail, 0ULL, (unsigned long long int)next_tail);

		}


		__device__ void detect_loop(){


			node_type * loop_one = tail;

			node_type * loop_two = tail->next;

			while(loop_two != nullptr && loop_one != nullptr){

				if (loop_one == loop_two){
					printf("Loop detected\n");
				}


				loop_one = tail->next;

				node_type * loop_two_next = loop_two->next;

				if (loop_two_next == nullptr) return;

				loop_two = loop_two_next->next;

			}


			printf("No loop detected\n");

		}

		__device__ node_type * trace_tail(node_type * my_tail){

			//node_type * my_tail = tail;

			while (my_tail->next != nullptr){


				my_tail = my_tail->next;


			}

			return my_tail;


		}

		__device__ bool enqueue(T new_item){

			node_type * my_tail = trace_tail(tail);

			node_type * original_tail = tail;
			

			//enqueue will succeed - only fail if malloc fails.
			while (true){


				detect_loop();

				__threadfence();

				my_tail = trace_tail(my_tail);

				//printf("Looping tail %lx\n", my_tail);

				//always at least one node to look at
				if (my_tail->enqueue(new_item)) return true;

				if (my_tail->next == nullptr){

					//printf("Mallocing new node\n");

					node_type * new_tail = (node_type *) gallatin::allocators::global_malloc(sizeof(node_type));

					printf("Exiting malloc\n");

					if (new_tail == nullptr) return false;

					new_tail->init();


					node_type * next = add_to_tail(&my_tail->next, new_tail);

					if (next == nullptr){

						my_tail = new_tail;
						
						//atomicCAS((unsigned long long int *)&tail, (unsigned long long int) original_tail, (unsigned long long int) new_tail);
						__threadfence();

					} else {
						my_tail = next;
						//printf("Freeing node\n");
						//gallatin::allocators::global_free(new_tail);
						__threadfence();
					}

					continue;
						
					}

				my_tail = my_tail->next;

				}



		}


		//valid to make optional type?

		// __device__ bool dequeue(T & return_val){

		// 	//do these reads need to be atomic? 
		// 	//I don't think so as these values don't change.
		// 	//as queue doesn't change ABA not possible.

		// 	__threadfence();

		// 	my_type * my_head = head;

		// 	if (head == nullptr){
		// 		return false;
		// 	}

		// 	my_type * head_next = my_head->next;

		// 	my_type * swap = (my_type *) atomicCAS((unsigned long long int *)&head, (unsigned long long int)my_head, (unsigned long long int)head_next);
			
		// 	while (swap != my_head){
		// 		__threadfence();

		// 		my_head = swap;

		// 		if (head == nullptr){
		// 			return false;
		// 		}


		// 		head_next = my_head->next;
		// 		swap = (my_type *) atomicCAS((unsigned long long int *)&head, (unsigned long long int)my_head, (unsigned long long int)head_next);
			


		// 	}

		// 	//at the end my_head is valid

		// 	__threadfence();

		// 	return_val = my_head->item;

		// 	//would be nice to have a hazard pointer here.

		// 	my_backing_allocator->free(my_head);

		// 	return true;

		// }




	};


}


}


#endif //end of queue name guard