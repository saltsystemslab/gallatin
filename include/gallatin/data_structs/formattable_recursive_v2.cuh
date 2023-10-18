#ifndef GALLATIN_FORMATTABLE_RECURSIVE_V2
#define GALLATIN_FORMATTABLE_RECURSIVE_V2


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
// #include <gallatin/allocators/murmurhash.cuh>

// #include <gallatin/data_structs/ds_utils.cuh>


#include <gallatin/data_structs/formattable_v2.cuh>




namespace gallatin {

namespace data_structs {

	//de-amortized formatted array
	//forces that the first time memory is observed it must be 0.
	//delayed global read allows for faster cached check on fast path.
	//stride sets granularity of calloc check.
	template <typename T, uint format_code=0U>
	struct formattable_recursive_alt {

		using my_type = formattable_recursive_alt<T, format_code>;

		T * data;

		formattable<uint64_t, ~0U> needs_flushed;
		formattable<uint64_t, ~0U> is_flushed;

		//how to perform swap
		//if resizing, 
		//perform global reads until new keys, vals are available.

		__device__ formattable_recursive_alt(uint64_t nitems): needs_flushed( (((nitems-1)/stride)/64)+1), is_flushed ((((nitems-1)/stride)/64)+1) {

			uint64_t total_bytes = nitems*sizeof(T);

			uint64_t total_flush_bits = (nitems-1)/stride+1;

			uint64_t total_flush_uint64_t = (total_flush_bits-1)/64+1;


			data = (T* ) gallatin::allocators::global_malloc(total_bytes);

			//needs_flushed = formattable<uint64_t, ~0U>(total_flush_uint64_t);

			//is_flushed = formattable<uint64_t, ~0U>(total_flush_uint64_t);

			// = (uint64_t *) gallatin::allocators::global_malloc(total_flush_uint64_t*sizeof(uint64_t));
			//is_flushed = (uint64_t *) gallatin::allocators::global_malloc(total_flush_uint64_t*sizeof(uint64_t));


			if (data == nullptr){
				//printf("Failed to malloc\n");

				asm volatile ("trap;");
			}

			//printf("Allocated: %lu items\n", nitems);

			//stride sets granularity of work.
			// for (uint64_t i = 0; i < total_flush_uint64_t; i++){

			// 	atomicExch((unsigned long long int *)&is_flushed[i], ~0ULL);
			// 	//is_flushed[i] = ~0ULL;
			// }

			//printf("Is flushed done, writing needs_flushed\n");

			// for (uint64_t i = 0; i < total_flush_uint64_t; i++){

			// 	atomicExch((unsigned long long int *)&needs_flushed[i], ~0ULL);
			// 	//needs_flushed[i] = ~0ULL;
			// }

			__threadfence();

			//at this point, you can point to calloc safely.


		}

		__device__ void print_calloc_status(uint64_t index){

			uint64_t bit = index/stride;

			uint64_t high = bit/64;

			//should be auto translated.
			uint64_t low = bit % 64;

			uint64_t calloced_bits = is_flushed[high];

			if (!(calloced_bits & SET_BIT_MASK(low))){
				//previously flushed

				printf("Flushed by earlier thread\n");
			}

			//otherwise, do a true global load to double triple check.

			calloced_bits = atomicAdd((unsigned long long int *)&is_flushed[high], 0ULL);


			//calloced_bits = gallatin::utils::ldca(&is_flushed[high]);

			if (!(calloced_bits & SET_BIT_MASK(low))){
				//previously flushed

				printf("Flushed by earlier thread, global load needed\n");
			}


			printf("Index %lu Needed atomic Unset check\n", index);


		}

		__device__ bool check_if_need_calloc(uint64_t index){

			uint64_t bit = index/stride;

			uint64_t high = bit/64;

			//should be auto translated.
			uint64_t low = bit % 64;

			uint64_t calloced_bits = is_flushed[high];

			if (!(calloced_bits & SET_BIT_MASK(low))){
				//previously flushed

				__threadfence();
				return false; 
			}

			//otherwise, do a true global load to double triple check.

			//calloced_bits = gallatin::utils::ldca(&is_flushed[high]);

			calloced_bits = atomicAdd((unsigned long long int *)&is_flushed[high], 0ULL);


			if (!(calloced_bits & SET_BIT_MASK(low))){
				//previously flushed
				__threadfence();
				return true;
			}

			//if we are here - need to see if we can acquire
			//opportunistic group form seems to avoid issues with 
			//local read vs atomic priority

			//it breaks without this...

    		//cg::coalesced_group coalesced_team = labeled_partition(full_warp_team, tree_id);

			if (atomicAnd((unsigned long long int *)&needs_flushed[high], ~SET_BIT_MASK(low)) & SET_BIT_MASK(low)){

				//printf("Starting region %lu calloc\n", bit);
				//I am responsible for updating!
				calloc_region(bit);

				//printf("Calloc %lu done\n", bit);

				//and write so that future threads observe calloc
				//this stalls when multiple threads in same warp want it.

				if(!(atomicAnd((unsigned long long int *)&is_flushed[high], ~SET_BIT_MASK(low)) & SET_BIT_MASK(low))){
					//this is bizarre behavior, implies that the region was previously calloced.

					//printf("Failed to set %lu bit %lu\n", high, low);
					asm("trap;");
				}

				__threadfence();

				//this doesn't trigger...
				//printf("Atomic write to %lu done\n", bit);

			}

			//calloc of region is either live or I finished


			while (calloced_bits & SET_BIT_MASK(low)){

				//calloced_bits = gallatin::utils::ldcg(&is_flushed[high]);

				calloced_bits = atomicAdd((unsigned long long int *)&is_flushed[high], 0ULL);

				//printf("Reading %lu\n", bit);
			}


			__threadfence();


			return true;

		}


		//write STRIDE*sizeof(T) bytes of memory to 0, based on the region bit.
		__device__ void calloc_region(uint64_t region_bit){

			// char * byte_start = (char *) &data[region_bit*stride];

			// for (uint64_t i = 0; i< stride*sizeof(T); i++){

			// 	//byte_start[i] = (char) 0;
			// 	//use atomic to set...
			// 	//gallatin::utils::global_store_byte(&byte_start[i], (char) 0);

			// }

			// __threadfence();


			uint * byte_start = (uint *) &data[region_bit*stride];

			//attempt to accelearate this via unroll
			#pragma unroll
			for (uint64_t i = 0; i < (stride*sizeof(T))/4; i++){

				//byte_start[i] = (char) 0;
				//use atomic to set...
				//gallatin::utils::global_store_byte(&byte_start[i], (char) 0);

				atomicExch(&byte_start[i], format_code);

			}

			__threadfence();

		
		}

		//deference operator - double check that memory has been cleared before giving back.
		__device__ T& operator[](uint64_t index)
		{
		    bool check_if_need_load = check_if_need_calloc(index);

		    //if (check_if_need_load){

		   	//gallatin::utils::ldcg(&data[index]);

		  	// } 

		   return data[index];
		   
		}

		//set memory just to assert that calloc works.
		//this may be messing things up by touching the memory early?
		//without this can get guarantee that memory has not been touched since it was allocated
		__device__ void debug_set_memory(uint64_t index, T item){


			return;

			data[index] = item;

			__threadfence();

		}


		//move constructor
		__device__ formattable_recursive_alt(formattable_recursive_alt&& other){

			//move pointers
			data = other.data;
			needs_flushed = other.needs_flushed;
			is_flushed = other.is_flushed;

			//clear pointers;
			other.data = nullptr;
			other.needs_flushed = nullptr;
			other.is_flushed = nullptr;

		}

		__device__ formattable_recursive_alt operator=(const formattable_recursive_alt & first){

			data = first.data;

			needs_flushed = first.needs_flushed;

			is_flushed = first.is_flushed;

		}

		__device__ void free_memory(){

			gallatin::allocators::global_free(data);

			is_flushed.free_memory();
			needs_flushed.free_memory();
			//gallatin::allocators::global_free(is_flushed);
			//gallatin::allocators::global_free(needs_flushed);

		}


		__device__ static my_type * get_pointer(uint64_t nitems){

			my_type * memory = (my_type *) gallatin::allocators::global_malloc(sizeof(my_type));

			memory[0] = my_type(nitems);

			return memory;

		}

	};


}


}


#endif //end of resizing_hash guard