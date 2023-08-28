#ifndef GALLATIN_VECTOR
#define GALLATIN_VECTOR


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <poggers/allocators/alloc_utils.cuh>


namespace gallatin {

namespace data_structs {


	#define VECTOR_LOCK_BIT 63
	#define VECTOR_LOCK_BITMASK SET_BIT_MASK(VECTOR_LOCK_BIT)

	#define TRANSMIT 62
	#define VECTOR_TRANSMIT_BITMASK SET_BIT_MASK(TRANSMIT)

	#define TRANSMIT_DONE_BIT 61
	#define VECTOR_TRANSMIT_DONE_BITMASK SET_BIT_MASK(TRANSMIT_DONE_BIT)

	#define VECTOR_COUNTER_BITS 36
	#define VECTOR_COUNTER_BITMASK (BITMASK(16) << VECTOR_COUNTER_BITS)

	//extra little debug to give a warning when the vector runs out of space
	//may help with debugging larger application.
	#define VECTOR_WARN_TOO_LARGE 0

	//turn on to give generic warnings when the vector fails an application
	//used for debugging the vector class, should not be enabled for production.
	#define VECTOR_WARN_DEBUG 0

	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current


	//Vector - a resizable templated container built for high performance in CUDA.
	
	//Features:
	//  -- configurable typing in a header-only format for fast execution
	//  -- additional workers are automatically pulled into the environment to assist resizes.
	//  == TODO: allow for dynamic parallelism if #workers crosses lower threshold?


	//T - type inserted and retreived from the vector
	//Allocator - allocator type - this is Gallatin normally, but let's expose it to other systems.
	//resizable - allow the vector to be resized 
	// - if not true insertions can fail instead of expanding the vector
	//    but space usage will not increase past the set size.
	template <typename T, typename allocator, bool resizable = true>
	struct vector {

		using my_type = vector<T, allocator, resizable>;


		//data counters contains:
		//max size
		//# of slots in use
		//#read bit
		//#use bit
		//#resize lock bit?

		//breakup- 2 bits up top, 16 bit counter, 4 bit pad, 32 bits
		//lots of space...
		uint64_t data_counters;

		//how many threads are in the reader/writer queues?
		//live threads *must* assist with resize.
		uint64_t readable_count;

		uint movement_counter;
		uint movement_finished_counter;

		T * data;

		//temporary pointer for managing incoming pointers
		//allows for any thread to contribute to the move
		T * new_data;

		allocator * my_backing_allocator;




		__device__ vector(uint64_t starting_count, allocator * external_alloc){

			my_backing_allocator = external_alloc;


			uint64_t first_bit_bigger = poggers::utils::get_first_bit_bigger(starting_count);
			uint64_t rounded_size = (1ULL << first_bit_bigger);

			data = (T * ) my_backing_allocator->malloc(sizeof(T)*rounded_size);

			new_data = nullptr;
			

			//setup data counters
			//lower 32 bits are atomicAdd counter - support size up to 4 billion.
			data_counters = (first_bit_bigger) << VECTOR_COUNTER_BITS;

			readable_count = 0;
			movement_counter = 0;


		}

		//delete memory used by this vector.
		//atm not really threadsafe so only do it if you're sure.
		__device__ void free_vector(){

			//swap to nullptr.
			T * ext_data = (T *) atomicExch((unsigned long long int *)&data, 0ULL);

			__threadfence();

			my_backing_allocator->free(ext_data);

		}

		//fail code is 64 1 bits.
		__device__ uint64_t fail(){
			return ~0ULL;
		}


		//called by all threads to check if they can assist the copy mechanism.
		//if locked, then perform the copy.
		__device__ void maybe_assist_copy(uint64_t old_counters, uint64_t current_size_bits, bool acquired_lock){


				//first, check if locked.
				if (old_counters & VECTOR_LOCK_BIT){


					//if locked, we should stall and wait for the load to occur.
					//can this resolve into an infinite loop?
					while (!(old_counters & VECTOR_TRANSMIT_BITMASK)){

						__threadfence();

						old_counters = poggers::utils::ldca((uint64_t *)&data);

					}


					//at this point the helper lock has been acquired.
					//all items must be ready for transfer

					uint64_t available_size = (1ULL << current_size_bits);

					uint64_t my_index = atomicAdd((unsigned int *)&movement_counter, 1U);
					while (my_index < available_size){

						__threadfence();

						new_data[my_index] = data[my_index];


						if (my_index == (available_size-1)){
							//set transfer done
							atomicOr((unsigned long long int *)&data_counters, VECTOR_TRANSMIT_DONE_BITMASK);

						}

						atomicAdd((unsigned int *)&movement_finished_counter, 1U);

						my_index = atomicAdd((unsigned int *)&movement_counter, 1U);
						

					}

					__threadfence();

					//responsible for marking finished.
					if (acquired_lock){

						uint64_t current_size = (1ULL << current_size_bits);

						uint64_t transfers_done = poggers::utils::ldca(&movement_finished_counter);

						while (transfers_done != current_size){

							transfers_done = poggers::utils::ldca(&movement_finished_counter);

						}


						__threadfence();

						T * old_data_ptr = data;

						data = new_data;

						__threadfence();
						//generate new vector control.

						uint64_t new_control =  (current_size_bits*2) << VECTOR_COUNTER_BITS;

						//we know for certain that exactly current_size items must have been inserted.
						new_control += current_size;

						atomicExch((unsigned long long int *)&data_counters, new_control);


						my_backing_allocator->free(old_data_ptr);

					}



				}


		}


		__device__ uint64_t insert(T item){




			//only loop when resizing - but if resizing we should always attempt
			while (true){

				//boolean flags for intra-thread ops

				//succeeded is true iff you successfully claim a spot - condition for returning
				bool succeeded = false;

				//acquired lock is true if you set the 
				bool acquired_lock = false;

				//first, get address to place item
				uint64_t old_counters = atomicAdd((unsigned long long int *)&data_counters, 1ULL);


				//if valid, can always proceed.
				uint64_t available_bits = old_counters & VECTOR_COUNTER_BITMASK;

				uint64_t available_size = (1ULL << available_bits);

				uint64_t my_count = old_counters & BITMASK(32);

				if (my_count < available_size){

					//proceed with insertion
					//At this point data is valid, so global load and insert

					T * local_data_ptr = (T *) poggers::utils::ldca((uint64_t *)&data);

					if (local_data_ptr == nullptr){
						return fail();
					}

					local_data_ptr[my_count] = item;

					__threadfence();

					//once changes are flushed, add to the counter to signal 
					//do we need a bitvector here?
					//this limits potential overwrites but does not entirely prevent them
					//unless this uses atomicCAS to swap from my_count to my_count+1.
					atomicAdd((unsigned long long int *)&readable_count, 1ULL);

					succeeded = true;


				} else if (!resizable){

					//if we can't resize and we can't claim space, fail.
					return fail();
				} else {


					//if you can see the lock is occupied, don't be the dumbass that attempts to acquire it.
					if (!(old_counters & VECTOR_LOCK_BIT)){

						//acquire permission to resize and signal to new threads that resize is immanent.
						acquired_lock = !(atomicOr((unsigned long long int *)&data_counters, VECTOR_LOCK_BIT) & VECTOR_LOCK_BIT);

					}

					if (acquired_lock){


						//uint64_t new_size = available_size*2

						T * new_vector = my_backing_allocator->malloc(sizeof(T)*available_size*2);

						if (new_vector == nullptr){

							#if VECTOR_WARN_TOO_LARGE

							printf("Vector of size %llu failed to expand to %llu\n, allocator returned nullptr", available_size, available_size*2);

							#endif

							//on failure, undo and return fail.

							atomicAnd((unsigned long long int *)&data_counters, ~VECTOR_LOCK_BIT);

							return fail();

						}


						//at this point we have a valid new pointer.

						//move to new pointer holding slot

						atomicExch((unsigned long long int *)&new_data, (unsigned long long int) new_vector);


						//this is a failure - should never be able to interrupt an in-progress transmission.
						if (new_data != nullptr){

							#if VECTOR_WARN_DEBUG

							printf("Error - Vector %llx overwrote existing new data pointer\n", (uint64_t) this);

							#endif

							asm("trap;");
						}


						__threadfence();


						uint64_t my_readable_count = poggers::utils::ldca(&readable_count);


						//all previous writes must complete before copy can occur
						//otherwise we risk moving nonexistant data.
						while (my_readable_count != available_size){

							__threadfence();
							my_readable_count = poggers::utils::ldca(&readable_count);

						}

						//set worker flags
						movement_counter = 0;
						movement_finished_counter = 0;

						__threadfence();

						//set transmit flag to signal that copy can begin

						//this must always be unset - at max one thread can enter.
						atomicOr((unsigned long long int *)&data_counters, VECTOR_TRANSMIT_BITMASK);

					}

				}


				//potentially participate in shared load
				maybe_assist_copy(acquired_lock);


				//at end of iteration, if we succeeded return
				if (succeeded){
					return my_count;
				}



			}

		}



	};


}


}


#endif //end of queue name guard