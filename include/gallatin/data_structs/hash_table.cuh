#ifndef GALLATIN_RESIZING_HASH
#define GALLATIN_RESIZING_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>



namespace gallatin {

namespace data_structs {

	#define GAL_QUAD_ASSIST_STRIDE 32
	#define GAL_QUAD_PROBE_DEPTH 25

	using namespace gallatin::allocators;

	//resizable quadratic probing table
	//This allows threads to progress on insertions, and redo work on resize
	template <typename Key, typename Val>
	struct quad_table {

		int resizing;
		uint64_t nslots;
		uint64_t seed;

		Key * keys;
		Val * vals;


		//counters control intro/exit of data movement.
		uint64_t next_nslots;
		uint64_t finished_move_nslots;
		uint64_t moved_nslots;
		Key * new_keys;
		Val * new_vals;

		//how to perform swap
		//if resizing, 
		//perform global reads until new keys, vals are available.

		__device__ void init(uint64_t initial_nslots=100, uint64_t ext_seed){

			keys = (Key *) global_malloc(initial_nslots*sizeof(Key));

			vals = (Val *) global_malloc(initial_nslots*sizeof(Val));

			seed = ext_seed;

			__threadfence();

		}


		//called by one thread - this triggers the resizing flag
		//then mallocs the buffers needed for the resize.
		__device__ void prep_upsize(){

			//first thread to do this triggers
			//then everyone waits on assert_key_vals_loaded
			if (atomicCAS((int *)&resizing, 0, 1) == 0){

				finished_move_nslots = 0;
				moved_nslots = 0;
				next_nslots = nslots*2;

				__threadfence();

				new_keys = (Key *) global_malloc(sizeof(Key)*next_nslots);
				new_vals = (Val *) global_malloc(sizeof(Val)*next_nslots);

				__threadfence();


			}


		}


		//when called, waits until new arrays are visible.
		__device__ void assert_keys_vals_loaded(){

			while(gallatin::utils::ldca((uint64_t *)&new_keys) == nullptr);

			while(gallatin::utils::ldca((uint64_t *)&new_vals) == nullptr);

		}


		__device__ void resize(){

			prep_upsize();

			assert_keys_vals_loaded();

			assist_with_copy();

		}

		//pull cooperative group for optimal memory access?
		//nit for now, maybe nice optimization.	
		__device__ void assist_with_copy(){

			uint64_t my_nslots = gallatin::utils::ldca(&nslots);
			uint64_t my_next_nslots = gallatin::utils::ldca(&next_nslots);

			while (true){

				//read a new batch of numbers to assist with copy
				uint64_t my_copy_start = atomicAdd((unsigned long long int *)&next_nslots, GAL_QUAD_ASSIST_STRIDE);

				uint64_t items_to_move;

				if (my_copy_start >= my_nslots){
					//all items already copied - end
					return;


				} else {
					uint64_t items_left = my_nslots - my_copy_start;

					if (items_left > GAL_QUAD_ASSIST_STRIDE){
						items_to_move = GAL_QUAD_ASSIST_STRIDE;
					} else {
						items_to_move = items_left;
					}

				}

				for (uint64_t i = 0; i < items_to_move; i++){

					uint64_t slot_index = (my_copy_start + i); // % my_nslots;

					Key copy_key = typed_atomic_exchange(&keys[slot_index], (Key) 0);

					//non-empty keys go to the new table.
					if (copy_key != (Key)0){

						//read and insert new val
						Val copy_val = typed_global_read(&vals[slot_index]);

						internal_insert_key_val_pair(new_keys, new_vals, copy_key, copy_val);

					}

				}				


			}

			


		}

		//actually insert
		//assumes as a precondition table is large enough
		//this returns true if success,
		//false if probe depth exceeded
		__device__ bool internal_insert_key_val_pair(Key * ext_keys, Val * ext_vals, uint64_t ext_nslots, Key key, Val val){


			uint64_t hash = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed);

			for (uint64_t i = 0; i < GAL_QUAD_PROBE_DEPTH; i++){

				uint64_t slot = (hash + i*i) % ext_nslots;

				if (ext_keys[slot] == (Key) 0){

					//maybe write!
					if (typed_atomic_write(&ext_keys[slot], (Key)0, key)){

						ext_vals[slot] = val;

						return true;

					}

				}

			}
			//all probes failed.
			return false;

		}

		//perform insertion, and then back up table
		__device__ void insert(Key key, Val val){


			while (true){

				//steps

				//1) if resize is visibly triggered, assist with load
				if (resizing){

					//load new key_val_pairs
					assert_keys_vals_loaded();

					assist_with_copy();

				}


			}




		}

	};


}


}


#endif //end of resizing_hash guard