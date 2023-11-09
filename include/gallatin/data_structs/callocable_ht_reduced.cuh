#ifndef GALLATIN_RESIZING_REDUCED_HASH
#define GALLATIN_RESIZING_REDUCED_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>

#include <gallatin/data_structs/callocable.cuh>

//#include <gallatin/data_structs/formattable.cuh>

#include <gallatin/data_structs/formattable_v2.cuh>


#define USE_ATOMICS 1

//atomic version of the table
//ldcg is for chumps...
//this is to verify correctness before moving to looser instructions.
namespace gallatin {

namespace data_structs {



	//helper kernel
	//each resize triggers a new kernel to insert items. 
	template <typename key_arr_type, typename val_arr_type, typename Table>
	__global__ void reinsert_old_keys(key_arr_type * old_keys, val_arr_type * old_vals, uint64_t nslots, Table * ext_table){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nslots) return;

		auto my_key = old_keys[0][tid];

		if (my_key == 0) return;

		auto my_val = old_vals[0][tid];

		ext_table->insert(my_key, my_val);


	}


	#define GAL_QUAD_ASSIST_STRIDE 32
	#define GAL_QUAD_PROBE_DEPTH 50
	#define GAL_QUAD_SIZE_START 50

	using namespace gallatin::allocators;
	using namespace gallatin::utils;

	template <typename Key_data_type, typename Val_data_type>
	struct combined_data_pointers {

		Key_data_type * keys;
		Val_data_type * vals;
		uint64_t merged_counter;


	};

	//resizable quadratic probing table
	//This allows threads to progress on insertions, and redo work on resize
	template <typename Key, typename Val, int stride = 1>
	struct quad_table {


		//resize strategy
		//one thread holds on to key/vals
		//attempt insert
		//if it fails, attempt resize
		//if that fails, wait on resize.

		using my_type = quad_table<Key, Val, stride>;

		
		//mixed counter for p2 of size
		//upper 14 bits reserved for counter.
		//with atomicExch to swap to new.
		uint64_t insert_size_counters;

		uint64_t seed;

		using key_arr_type = formattable_alt<Key, (Key) 0>;
		using val_arr_type = formattable_alt<Val, (Key) 0>;

		using combined_type = combined_data_pointers<key_arr_type, val_arr_type>;

		combined_type * combined_data;

		// using key_arr_type = callocable<Key>;
		// using val_arr_type = callocable<Val>;

		//key_arr_type * keys;
		//val_arr_type * vals;


		//counters control intro/exit of data movement.
		uint64_t next_nslots;


		//data copy values
		uint64_t finished_move_nslots;
		uint64_t moved_nslots;

		//new array values
		uint64_t clear_nslots;
		uint64_t finished_clear_nslots;

		uint64_t finished_counter;

		uint assisting_resize;

		double resize_ratio;


		//how to perform swap
		//if resizing, 
		//perform global reads until new keys, vals are available.

		__device__ void init(uint64_t initial_nslots=100, uint64_t ext_seed=4095, double ext_resize_ratio=.77){


			uint init_bits = gallatin::utils::get_first_bit_bigger(initial_nslots);

			initial_nslots = 1ULL << init_bits;

			printf("New size is %lu\n", initial_nslots);

			//keys = (Key *) gallatin::allocators::global_malloc(sizeof(Key)*initial_nslots);

			//vals = (Val *) gallatin::allocators::global_malloc(sizeof(Val)*initial_nslots); 

			//gallatin::utils::memclear(keys, initial_nslots, initial_nslots/32);
			//gallatin::utils::memclear(vals, initial_nslots, initial_nslots/32);

			auto keys = key_arr_type::get_pointer(initial_nslots);
			auto vals = val_arr_type::get_pointer(initial_nslots);



					//build new stuff
			combined_type * new_data_pointers = (combined_type *) gallatin::allocators::global_malloc(sizeof(combined_type));

			uint64_t new_counter = ((uint64_t) init_bits) << 50;

			new_data_pointers->keys = keys;
			new_data_pointers->vals = vals;
			new_data_pointers->merged_counter = new_counter;

			__threadfence();

			combined_data = new_data_pointers;

			//non global init.
			// for (uint64_t i = 0; i < initial_nslots; i++){

			// 	keys[0][i] = 0;
			// 	vals[0][i] = 0;

			// }

			seed = ext_seed;

			resize_ratio = ext_resize_ratio;

			

			finished_counter = 0;


			__threadfence();

		}


		__device__ uint64_t get_nslots(uint bits){
			return 1ULL << bits;
		}

		__device__ uint64_t get_fill(uint64_t counter){

			return counter & BITMASK(50);

		}

		__device__ uint get_bits(uint64_t counter){

			return counter >> 50;

		}


		//actually insert
		//assumes as a precondition table is large enough
		//this returns true if success,
		//false if probe depth exceeded
		__device__ int internal_insert_key_val_pair(key_arr_type * ext_keys, val_arr_type * ext_vals, uint64_t ext_nslots, Key key, Val val){


			uint64_t hash = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed);

			for (uint64_t i = 0; i < GAL_QUAD_PROBE_DEPTH; i++){

				uint64_t slot = (hash + i*i) % ext_nslots;

			

				//maybe write!


				if (ext_keys->wrappedAtomicCAS(slot, (Key)0, key) == (Key) 0){
				//if (typed_atomic_write(&ext_keys[0][slot], (Key)0, key)){

					ext_vals->wrappedAtomicExch(slot, val);

					return i;

				}

				

			}
			//all probes failed.
			return GAL_QUAD_PROBE_DEPTH;

		}

		//perform insertion, and then back up table if something happened.
		__device__ void insert(Key key, Val val){



			key_arr_type * my_keys;
			val_arr_type * my_vals;


			while (true){

				
				combined_type * atomic_read_address = (combined_type *) atomicAdd((unsigned long long int *)&combined_data, 0ULL);

				if (atomic_read_address == nullptr){

					asm volatile("trap;");
					return;
				}

				uint64_t merged_counter = atomicAdd((unsigned long long int *)&atomic_read_address->merged_counter, 1ULL);
					

				uint bits = get_bits(merged_counter);
				uint64_t my_nslots = get_nslots(bits);
				uint64_t old_count = get_fill(merged_counter);

				uint64_t my_count = old_count+1;

				uint64_t resize_threshold = my_nslots*resize_ratio;


				//trigger to produce a new array - 

				if (my_count >= resize_threshold && old_count < resize_threshold){


					//printf("Thread is in charge of resize! %lu vs %lu\n", my_count, resize_threshold);

					atomicAdd((unsigned long long int*)&finished_counter, 1ULL);

					while (atomicAdd((unsigned long long int* )&finished_counter, 0ULL) < resize_threshold);

					
					//reset to 0.
					atomicExch((unsigned long long int* )&finished_counter, 0ULL);


					uint64_t next_nslots = (1ULL << (bits+1));
					//printf("Resize progressed! %lu new size\n", next_nslots);

					key_arr_type * new_keys = key_arr_type::get_pointer(next_nslots);
					val_arr_type * new_vals = val_arr_type::get_pointer(next_nslots);

					my_keys = atomic_read_address->keys;
					my_vals = atomic_read_address->vals;

					// my_keys = (key_arr_type *) atomicAdd((unsigned long long int *)&keys, 0ULL);
					// my_vals = (val_arr_type *) atomicAdd((unsigned long long int *)&vals, 0ULL);


					//swap_to_new_array(keys, new_keys);
					//swap_to_new_array(vals, new_vals);

					//build new stuff
					combined_type * new_data_pointers = (combined_type *) gallatin::allocators::global_malloc(sizeof(combined_type));

					uint64_t new_counter = ((uint64_t) (bits+1)) << 50;

					new_data_pointers->keys = new_keys;
					new_data_pointers->vals = new_vals;
					new_data_pointers->merged_counter = new_counter;

					__threadfence();


					atomicExch((unsigned long long int *)&combined_data, (unsigned long long int)new_data_pointers);


					__threadfence();
					//atomicExch((unsigned long long int *)&insert_size_counters, (unsigned long long int)new_counter);

					//printf("Resize done, new size: %lu\n", 1ULL << (bits+1));

					//trigger resize kernel here.
					reinsert_old_keys<key_arr_type, val_arr_type, my_type><<<(my_nslots-1)/256+1,256>>>(my_keys, my_vals, my_nslots, this);

					//and re-enter the loop
					continue;

				} else if (my_count >= resize_threshold){

					//wait on resize

					combined_type * new_bits;


					do {

						new_bits = (combined_type *) atomicAdd((unsigned long long int *)&combined_data, 0ULL);



					} while (new_bits == atomic_read_address);

					//resize done! exit loop
					continue;



				} else {

					//precondition that pointers do not change for lifetime.
					//this lazy read is ok.

					my_keys = atomic_read_address->keys;
					my_vals = atomic_read_address->vals;


					int result = internal_insert_key_val_pair(my_keys, my_vals, my_nslots, key, val);


					#if USE_ATOMICS

					atomicAdd((unsigned long long int *)&finished_counter, 1ULL);

					#endif


					if (result == GAL_QUAD_PROBE_DEPTH){
						//printf("Failed to insert %lu\n", key);

						continue;
					}

					return;

				}



			}

		}
	


		// 		//resizing set to false now - try insert on new keys/vals

		// 		uint64_t * addr_of_keys = (uint64_t *) &keys;

		// 		Key * local_keys = (Key *) gallatin::utils::ldcg(addr_of_keys);

		// 		uint64_t * addr_of_vals = (uint64_t *) &vals;

		// 		Val * local_vals = (Val *) gallatin::utils::ldcg(addr_of_vals);

		// 		uint64_t local_nslots = typed_global_read(&nslots);


		// 		int insert_slot = internal_insert_key_val_pair(local_keys, local_vals, local_nslots, key, val);

		// 		if (insert_slot == GAL_QUAD_PROBE_DEPTH){
		// 			//fail! resize
		// 			resize();
		// 		} else {

		// 			//succeeded.
		// 			return;
		// 		}


		// 		//full_warp_team.sync();
		// 		//potential error - resize could trigger while I was working.
		// 		//meaning that my table was written before I could fail?
		// 		//deal with this later - get test liv


		// 		//printf("Looping main loop\n");


		//can get away with one read to old state ?
		__device__ bool query(Key key, Val & val){


			//uint64_t merged_counter = atomicAdd((unsigned long long int *)&insert_size_counters, 1ULL);


			//combined_type * atomic_read_address = (combined_type *) atomicAdd((unsigned long long int *)&combined_data, 0ULL);

			auto keys = combined_data->keys;
			auto vals = combined_data->vals;


			uint64_t merged_counter = insert_size_counters;

			uint bits = get_bits(merged_counter);
			uint64_t nslots = get_nslots(bits);

			uint64_t hash = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed);

			for (uint64_t i = 0; i < GAL_QUAD_PROBE_DEPTH; i++){

				uint64_t slot = (hash + i*i) % nslots;


				if (keys[0][slot] == key){
					val = vals[0][slot];
					return true;
				}

				

			}


			return false;




		}

	};


}


}


#endif //end of resizing_hash guard