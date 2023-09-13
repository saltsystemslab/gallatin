#ifndef GALLATIN_RESIZING_HASH
#define GALLATIN_RESIZING_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>



namespace gallatin {

namespace data_structs {

	#define GAL_QUAD_ASSIST_STRIDE 32

	using namespace gallatin::allocators;

	//resizable quadratic probing table
	//This allows threads to progress on insertions, and redo work on resize
	template <typename Key, typename Val>
	struct quad_table {

		int resizing;
		uint64_t nslots;

		Key * keys;
		Val * vals;


		//counters control intro/exit of data movement.
		uint64_t finished_move_nslots;
		uint64_t moved_nslots;
		Key * new_keys;
		Val * new_vals;

		//how to perform swap
		//if resizing, 
		//perform global reads until new keys, vals are available.

		__device__ void init(uint64_t initial_nslots=100){

			keys = (Key *) global_malloc(initial_nslots*sizeof(Key));

			vals = (Val *) global_malloc(initial_nslots*sizeof(Val));

		}


		//called by one thread - this triggers the resizing flag
		//then mallocs the buffers needed for the resize.
		__device__ void prep_upsize(){



		}


		//when called, waits until new arrays are visible.
		__device__ void assert_keys_vals_loaded(){

			while(gallatin::utils::ldca((uint64_t *)&new_keys) == nullptr);

			while(gallatin::utils::ldca((uint64_t *)&new_vals) == nullptr);

		}

		//pull cooperative group for optimal memory access?
		//nit for now, maybe nice optimization.	
		__device__ void assist_with_copy(){


			uint64_t my_copy_start = atomicAdd(())


		}

		//perform insertion, and then back up table
		__device__ void insert(Key key, Val val){


			while (true){

				//steps

				//1) if resize is visibly triggered, assist with load
				if (resizing){

					assert_keys_vals_loaded();

					assist_with_copy();

				}


			}




		}

	};


}


}


#endif //end of resizing_hash guard