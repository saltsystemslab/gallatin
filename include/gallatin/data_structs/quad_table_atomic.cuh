#ifndef GALLATIN_RESIZING_HASH
#define GALLATIN_RESIZING_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>

#include <gallatin/data_structs/callocable.cuh>



//atomic version of the table
//ldcg is for chumps...
//this is to verify correctness before moving to looser instructions.
namespace gallatin {

namespace data_structs {


	#define GAL_QUAD_ASSIST_STRIDE 32
	#define GAL_QUAD_PROBE_DEPTH 50

	using namespace gallatin::allocators;
	using namespace gallatin::utils;

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


		//data copy values
		uint64_t finished_move_nslots;
		uint64_t moved_nslots;

		//new array values
		uint64_t clear_nslots;
		uint64_t finished_clear_nslots;


		Key * new_keys;
		Val * new_vals;

		//how to perform swap
		//if resizing, 
		//perform global reads until new keys, vals are available.

		__device__ void init(uint64_t initial_nslots=100, uint64_t ext_seed=4095){

			keys = (Key *) gallatin::allocators::global_malloc(sizeof(Key)*initial_nslots);

			vals = (Val *) gallatin::allocators::global_malloc(sizeof(Val)*initial_nslots); 

			//gallatin::utils::memclear(keys, initial_nslots, initial_nslots/32);
			//gallatin::utils::memclear(vals, initial_nslots, initial_nslots/32);

			//non global init.
			for (uint64_t i = 0; i < initial_nslots; i++){

				keys[i] = 0;
				vals[i] = 0;

			}

			new_keys = keys;
			new_vals = vals;

			seed = ext_seed;
			nslots = initial_nslots;

			__threadfence();

		}


		//called by one thread - this triggers the resizing flag
		//then mallocs the buffers needed for the resize.
		__device__ void prep_upsize(){

			//first thread to do this triggers
			//then everyone waits on assert_key_vals_loaded
			if (atomicCAS((int *)&resizing, 0, 1) == 0){

				//printf("Starting resize: %llu\n", nslots*2);

				typed_atomic_exchange(&finished_move_nslots, (uint64_t) 0ULL);
				typed_atomic_exchange(&moved_nslots, (uint64_t) 0ULL);
				typed_atomic_exchange(&next_nslots,  nslots*2);

				typed_atomic_exchange(&clear_nslots, (uint64_t) 0ULL);
				typed_atomic_exchange(&finished_clear_nslots, (uint64_t) 0ULL);

				__threadfence();

				//printf("Gathering new memory with size %lu->%lu\n", nslots, next_nslots);

				Key * temp_newkeys = (Key *) gallatin::allocators::global_malloc(sizeof(Key)*next_nslots);
				//Key::get_pointer(next_nslots); 

				Val * temp_newvals = (Val *) gallatin::allocators::global_malloc(sizeof(Val)*next_nslots);
		
				//Val * temp_newvals = (Val *) global_malloc(sizeof(Val)*next_nslots);

				// gallatin::utils::memclear(temp_newkeys, next_nslots, 10000);
				// gallatin::utils::memclear(temp_newvals, next_nslots, 10000);

				//this is freaking cursed
				//does this work? who knows...
				//this may trigger a failure.
				//ht_upsize_tail<Key, Val><<<1, 1, 0, cudaStreamTailLaunch>>>(&new_keys, temp_newkeys, &new_vals, temp_newvals);

				//atomic pointer swap these bad boys.

				swap_to_new_array(new_keys, temp_newkeys);

				swap_to_new_array(new_vals, temp_newvals);

				__threadfence();

				//printf("upsize done!\n");


			}


		}

		//spin on resizing until final control thread signals done.
		__device__ void finish_upsize(){


			while (atomicAdd(&resizing, 0) == 1);
			//while (typed_global_read(&resizing));


		}


		//when called, waits until new arrays are visible.
		__device__ void assert_keys_vals_loaded(){

			//printf("Checking key array\n");

			uint64_t * addr_of_new_keys = (uint64_t *) &new_keys;
			uint64_t * addr_of_new_vals = (uint64_t *) &new_vals;

			while(((Key *) gallatin::utils::ldcg(addr_of_new_keys)) == keys){

				addr_of_new_keys = (uint64_t *) &new_keys;
				//printf("Spinning\n");

			}

			while(((Key *) gallatin::utils::ldcg(addr_of_new_vals)) == vals){
				addr_of_new_vals = (uint64_t *) &new_vals;
			}

			//printf("New keys have been read\n");

		}


		__device__ void resize(){

			

			prep_upsize();

			assert_keys_vals_loaded();

			assist_with_copy();

			finish_upsize();

		}

		//pull cooperative group for optimal memory access?
		//nit for now, maybe nice optimization.	
		__device__ void assist_with_copy(){

			uint64_t my_nslots = gallatin::utils::ldcg(&nslots);
			uint64_t my_next_nslots = gallatin::utils::ldcg(&next_nslots);




			//start clear of data

			while (true){


				uint64_t my_copy_start = atomicAdd((unsigned long long int *)&clear_nslots, GAL_QUAD_ASSIST_STRIDE);

				uint64_t items_to_move;

				if (my_copy_start >= my_nslots){
					//all items already copied - end

					items_to_move = 0;

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

					Key copy_key = typed_atomic_exchange(&new_keys[slot_index], (Key) 0);


				}

				uint64_t copied_so_far = atomicAdd((unsigned long long int *)&finished_clear_nslots, items_to_move);

				//printf("Copied so far: %lu + %lu out of %lu: %f\n", copied_so_far, items_to_move, my_nslots, 1.0*(copied_so_far+items_to_move)/my_nslots);

				if ( (copied_so_far + items_to_move) == my_nslots) break;

					

			}

			//printf("Done with clear\n");



			//start copy
			while (true){

				//printf("Starting copy iteration\n");

				//cg::coalesced_group full_warp_team = cg::coalesced_threads();

				

				//read a new batch of numbers to assist with copy
				uint64_t my_copy_start = atomicAdd((unsigned long long int *)&moved_nslots, GAL_QUAD_ASSIST_STRIDE);

				uint64_t items_to_move;

				if (my_copy_start >= my_nslots){
					//all items already copied - end

					items_to_move = 0;

				} else {
					uint64_t items_left = my_nslots - my_copy_start;

					

					if (items_left > GAL_QUAD_ASSIST_STRIDE){
						items_to_move = GAL_QUAD_ASSIST_STRIDE;
					} else {
						items_to_move = items_left;
					}

				}

				//printf("Moving %lu items from %lu->%lu\n", items_to_move, my_copy_start, my_copy_start+items_to_move);

				for (uint64_t i = 0; i < items_to_move; i++){



					uint64_t slot_index = (my_copy_start + i); // % my_nslots;

					Key copy_key = typed_atomic_exchange(&keys[slot_index], (Key) 0);

					//non-empty keys go to the new table.
					if (copy_key != (Key)0){

						//read and insert new val
						Val copy_val = typed_global_read(&vals[slot_index]);

						auto num_copy_write = internal_insert_key_val_pair(new_keys, new_vals, my_next_nslots, copy_key, copy_val);

						//auto num_copy_write = 0;
						if (num_copy_write == GAL_QUAD_PROBE_DEPTH){
							printf("Failed to write!\n");
						}

					}

					//printf("Done with %lu/%lu\n", i, items_to_move);

				}

				//printf("Done moving %lu items\n", items_to_move);

				//register that items_to_move items have been copied

				if (items_to_move != 0){

					uint64_t copied_so_far = atomicAdd((unsigned long long int *)&finished_move_nslots, items_to_move);

					//printf("Copied so far: %lu + %lu out of %lu: %f\n", copied_so_far, items_to_move, my_nslots, 1.0*(copied_so_far+items_to_move)/my_nslots);

					if ( (copied_so_far + items_to_move) == my_nslots){

						//thread responsible for ending!
						//we know old array isn't needed, so copy!

						//typed_atomic_exchange((uint64_t *) &keys, (uint64_t) new_keys);
						//typed_atomic_exchange((uint64_t *) &vals, (uint64_t) new_vals);

						//printf("Ending copy: %lu vs %lu\n", copied_so_far+items_to_move, my_nslots);

						swap_to_new_array(keys, new_keys);
						swap_to_new_array(vals, new_vals);


						typed_atomic_exchange(&nslots, my_nslots*2);
						__threadfence();

						atomicCAS((int *)&resizing, 1, 0);

					}
				
				}

				//full_warp_team.sync();

				if (items_to_move == 0){

					//printf("Exiting copy: %lu > %lu\n", my_copy_start, my_nslots);

					return;

				}




			}

			


		}

		//actually insert
		//assumes as a precondition table is large enough
		//this returns true if success,
		//false if probe depth exceeded
		__device__ int internal_insert_key_val_pair(Key * ext_keys, Val * ext_vals, uint64_t ext_nslots, Key key, Val val){


			uint64_t hash = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed);

			for (uint64_t i = 0; i < GAL_QUAD_PROBE_DEPTH; i++){

				uint64_t slot = (hash + i*i) % ext_nslots;

			

				//maybe write!
				if (typed_atomic_write(&ext_keys[slot], (Key)0, key)){

					ext_vals[slot] = val;

					return i;

				}

				

			}
			//all probes failed.
			return GAL_QUAD_PROBE_DEPTH;

		}

		//perform insertion, and then back up table if something happened.
		__device__ void insert(Key key, Val val){


			while (true){

				//steps

				cg::coalesced_group full_warp_team = cg::coalesced_threads();

				//1) if resize is visibly triggered, assist with load!
				if (typed_global_read(&resizing)){

					//load new key_val_pair arrays
					assert_keys_vals_loaded();

					//start copy
					assist_with_copy();

					//copy is done!
					finish_upsize();

				}


				//resizing set to false now - try insert on new keys/vals

				uint64_t * addr_of_keys = (uint64_t *) &keys;

				Key * local_keys = (Key *) gallatin::utils::ldcg(addr_of_keys);

				uint64_t * addr_of_vals = (uint64_t *) &vals;

				Val * local_vals = (Val *) gallatin::utils::ldcg(addr_of_vals);

				uint64_t local_nslots = typed_global_read(&nslots);


				int insert_slot = internal_insert_key_val_pair(local_keys, local_vals, local_nslots, key, val);

				if (insert_slot == GAL_QUAD_PROBE_DEPTH){
					//fail! resize
					resize();
				} else {

					//succeeded.
					return;
				}


				full_warp_team.sync();
				//potential error - resize could trigger while I was working.
				//meaning that my table was written before I could fail?
				//deal with this later - get test live


			}




		}

		//can get away with one read to old state ?
		__device__ bool query(Key key, Val & val){

			uint64_t hash = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed);

			for (uint64_t i = 0; i < GAL_QUAD_PROBE_DEPTH; i++){

				uint64_t slot = (hash + i*i) % nslots;


				if (keys[slot] == key){
					val = vals[slot];
					return true;
				}

				

			}


			return false;




		}

	};


}


}


#endif //end of resizing_hash guard