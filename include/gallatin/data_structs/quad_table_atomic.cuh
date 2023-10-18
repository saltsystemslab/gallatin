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



	//helper kernel
	//each resize triggers a new kernel to insert items. 
	template <typename Key, typename Val, typename Table>
	__global__ void reinsert_old_keys(Key * old_keys, Val * old_vals, uint64_t nslots, Table * ext_table){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nslots) return;

		Key my_key = old_keys[tid];

		if (my_key == 0) return;

		Val my_val = old_vals[tid];

		ext_table->insert(my_key, my_val);


	}


	#define GAL_QUAD_ASSIST_STRIDE 32
	#define GAL_QUAD_PROBE_DEPTH 50

	using namespace gallatin::allocators;
	using namespace gallatin::utils;

	//resizable quadratic probing table
	//This allows threads to progress on insertions, and redo work on resize
	template <typename Key, typename Val>
	struct quad_table {


		//resize strategy
		//one thread holds on to key/vals
		//attempt insert
		//if it fails, attempt resize
		//if that fails, wait on resize.

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

		uint assisting_resize;


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

			new_keys = nullptr;
			new_vals = nullptr;

			seed = ext_seed;
			nslots = initial_nslots;

			next_nslots = nslots;

			assisting_resize = 0U;

			__threadfence();

		}


		//called by one thread - this triggers the resizing flag
		//then mallocs the buffers needed for the resize.
		__device__ int prep_upsize(uint64_t my_nslots){


			//start with conversion of my_next_nslots
			if (atomicCAS((unsigned long long int *)&next_nslots, my_nslots, my_nslots*2) == my_nslots){



			} else {

				//fail flag - tried to resize but it already happened
				return 0;

			}

			int my_resize_val = atomicCAS((int *)&resizing, 0, 1);

			//first thread to do this triggers
			//then everyone waits on assert_key_vals_loaded
			if (my_resize_val == 0){

				//printf("Starting resize: %llu\n", nslots*2);

				uint64_t old_next_nslots = atomicCAS((unsigned long long int *)&next_nslots, nslots, 2*nslots);

				if (old_next_nslots != nslots){

					atomicCAS((int *)&resizing, 1, 0);

					//printf("Reading from old upsize: was %lu\n", old_next_nslots);

					return 0;
				}


				typed_atomic_exchange(&finished_move_nslots, (uint64_t) 0ULL);
				typed_atomic_exchange(&moved_nslots, (uint64_t) 0ULL);
				

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


			return my_resize_val;


		}

		//spin on resizing until final control thread signals done.
		__device__ void finish_upsize(){


			while (atomicAdd(&resizing, 0) == 1);
			//while (typed_global_read(&resizing));


		}


		//when called, waits until new arrays are visible.
		__device__ void assert_keys_vals_loaded(Key * &my_new_keys, Val * &my_new_vals){

			//printf("Checking key array\n");

			//uint64_t * addr_of_new_keys = (uint64_t *) &new_keys;
			//uint64_t * addr_of_new_vals = (uint64_t *) &new_vals;

			my_new_keys = (Key *) atomicAdd((unsigned long long int *)&new_keys, 0ULL);

			while(my_new_keys == nullptr){

				my_new_keys = (Key *) atomicAdd((unsigned long long int *)&new_keys, 0ULL);



				//printf("Spinning\n");

				//printf("Looping on keys load.\n");

			}


			my_new_vals = (Val *) atomicAdd((unsigned long long int *)&new_vals, 0ULL);


			while (my_new_vals == nullptr){

				my_new_vals = (Val *) atomicAdd((unsigned long long int *)&new_vals, 0ULL);

			}

			// while(((Key *) gallatin::utils::ldcg(addr_of_new_vals)) == vals){
			// 	addr_of_new_vals = (uint64_t *) &new_vals;

			// 	printf("Looping on vals load.\n");
			// }

			//printf("New keys have been read\n");

		}


		__device__ void resize(Key * &my_new_keys, Val * &my_new_vals){

			

			if (!prep_upsize()) return;

			assert_keys_vals_loaded(my_new_keys, my_new_vals);

			assist_with_copy(my_new_keys, my_new_vals);

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

				if ( (copied_so_far + items_to_move  == my_nslots)){

					if (items_to_move != 0){
						//take control and move


						//move control flag to 2 - signals that new threads cannot enter
						atomicAdd(&resizing, 1);

						atomicSub((&assisting_resize, 1));

						//loop until all helpers have exited
						while (atomicAdd((unsigned int *)&assisting_resize, 0) != 0);

						//Key * old_keys = keys;
						//Val * old_vals = vals;

						swap_to_new_array(keys, my_new_keys);
						swap_to_new_array(vals, my_new_keys);


						swap_to_new_array(new_keys, nullptr);
						swap_to_new_array(new_vals, nullptr);

						//after this, reset is done - signal end by resetting to 0.
						atomicSub(&resizing, 2);

						

						printf("Resize done\n");


					}

					break;

				}


					

			}

			//printf("Done with clear\n");


			int my_resizing = 1;

			while (my_resizing){
				my_resizing = gallatin::utils::ldcg(&resizing);

				//printf("Looping on resize check.\n");
			}


			//printf("Resize done visible\n");

			return;

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


				//start with load of all 3 variables

				Key * my_new_keys = nullptr;
				Val * my_new_vals = nullptr;

				
				uint64_t my_nslots = atomicAdd((unsigned long long int *)&nslots, 0ULL);
				Key * my_keys = (Key *) atomicAdd((unsigned long long int *)&keys, 0ULL);
				Val * my_vals = (Val *) atomicAdd((unsigned long long int *)&vals, 0ULL);
				int my_resizing = atomicAdd(&resizing, 0);

				//flags
				//resizing == 0 - nothing going on
				//resizing == 1 - resizing started but not done
				//resizing == 2 - resizing finished.
				if (my_resizing == 1){


					atomicAdd((unsigned int *)&assisting_resize, 1U);

					//overwrite keys 
					assert_keys_vals_loaded(my_new_keys, my_new_vals);

					assist_with_copy(my_new_keys, my_new_vals);

					//signal # threads waiting on resource

					finish_upsize();

					atomicSub((unsigned int *)&assisting_resize, 1U);


					continue;

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


				//full_warp_team.sync();
				//potential error - resize could trigger while I was working.
				//meaning that my table was written before I could fail?
				//deal with this later - get test liv


				//printf("Looping main loop\n");


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