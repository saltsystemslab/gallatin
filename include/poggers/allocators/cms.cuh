#ifndef CMS
#define CMS


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/aligned_stack.cuh>
#include <poggers/allocators/sub_allocator.cuh>

//new allocator with dead list support
#include <poggers/allocators/dead_list_sub_allocator.cuh>

#include "stdio.h"
#include "assert.h"


//include files for the hash table
//hash scheme
#include <poggers/hash_schemes/murmurhash.cuh>

//probing scheme
#include <poggers/probing_schemes/double_hashing.cuh>

//insert_scheme
#include <poggers/insert_schemes/bucket_insert.cuh>

//table type
#include <poggers/tables/base_table.cuh>

//storage containers for keys
#include <poggers/representations/dynamic_container.cuh>
#include <poggers/representations/key_only.cuh>

//sizing type for building the table
#include <poggers/sizing/default_sizing.cuh>


//additional file for monitoring state of the system
#include <poggers/allocators/reporter.cuh>


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//CMS: The CUDA Memory Shibboleth
//CMS is a drop-in replacement for cudaMalloc and CudaFree.
//Before your kernels start, initialize a handler with
//shibboleth * manager = poggers::allocators::shibboleth::init() or init_managed() for host-device unified memory.
// The amount of memory specified at construction is all that's available to the manager,
// so you can spin up multiple managers for different tasks or request all available memory!
// The memory returned is built off of cudaMalloc or cudaMallocManaged, so regular cuda API calls are fine.
// Unlike the cuda device API, however, you can safely cudaMemcpy to and from memory requested by threads!



#if COUNTING_CYCLES
#include <poggers/allocators/cycle_counting.cuh>
#endif




//a pointer list managing a set section o fdevice memory

// 	const float log_of_size = std::log2()

// }

namespace poggers {


namespace allocators { 


template <typename allocator>
__global__ void one_thread_report_kernel(allocator * my_allocator){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   my_allocator->one_thread_report();

   return;

}

template <typename allocator>
__host__ void host_report_wrapper(allocator * my_allocator){

	one_thread_report_kernel<allocator><<<1,1>>>(my_allocator);

}

template <typename allocator>
__global__ void allocate_sub_allocator(allocator ** stack_ptr, header * heap){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   allocator * my_stack = allocator::init(heap);

   stack_ptr[0] = my_stack;

   return;

}

template <typename allocator>
__host__ allocator * host_allocate_sub_allocator(header * heap){

   allocator ** stack_ptr;

   cudaMallocManaged((void **)&stack_ptr, sizeof(allocator *));

   allocate_sub_allocator<allocator><<<1,1>>>(stack_ptr, heap);

   cudaDeviceSynchronize();

   allocator * to_return = stack_ptr[0];

   cudaFree(stack_ptr);

   return to_return;



}

template <typename cms>
__global__ void host_malloc_kernel(void ** stack_ptr, cms * shibboleth, uint64_t num_bytes){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   stack_ptr[0] = shibboleth->cms_malloc(num_bytes);

   //stack_ptr[0] = my_stack;

   return;

}

template <typename cms>
__host__ void * host_malloc_wrapper(cms * shibboleth, uint64_t num_bytes){

   void ** stack_ptr;

   cudaMallocManaged((void **)&stack_ptr, sizeof(void *));

   host_malloc_kernel<cms><<<1,1>>>(stack_ptr, shibboleth, num_bytes);

   cudaDeviceSynchronize();

   void * to_return = stack_ptr[0];

   cudaFree(stack_ptr);

   return to_return;


}

template <typename cms>
__global__ void host_free_kernel(cms * shibboleth, void * to_free){
	shibboleth->cms_free(to_free);
}

template <typename cms>
__host__ void host_free_wrapper(cms * shibboleth, void * to_free){

	host_free_kernel<cms><<<1,1>>>(shibboleth, to_free);

}


template <std::size_t bytes_per_substack, std::size_t num_suballocators, std::size_t maximum_p2>
struct shibboleth {


	using stack_type = aligned_manager<bytes_per_substack, false>;
	using my_type = shibboleth<bytes_per_substack, num_suballocators, maximum_p2>;

	using allocator = dead_list_sub_allocator<bytes_per_substack, maximum_p2>;
	//using allocator = sub_allocator<bytes_per_substack, maximum_p2>;


	using heap = header;

	using hash_table = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint64_t>::representation, 1, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

	allocator * allocators[num_suballocators];

	heap * allocated_memory;

	hash_table * free_table;




	static __host__ my_type * init_backend(std::size_t bytes_requested, bool managed){

		//this is gonna be very cheeky
		//an allocator can be hosted on its own memory!

		#if COUNTING_CYCLES
		poggers_reset_cycles();
		#endif

		my_type * host_cms = (my_type * ) malloc(sizeof(my_type)); 

		//allocated_memory;

		if (managed){
			host_cms->allocated_memory = heap::init_heap_managed(bytes_requested);
		} else {
			host_cms->allocated_memory = heap::init_heap(bytes_requested);
		}


		if (host_cms->allocated_memory == nullptr){
			printf("Failed to allocate device memory!\n");
			abort();
		}

		for (int i = 0; i < num_suballocators; i++){
			host_cms->allocators[i] = host_allocate_sub_allocator<allocator>(host_cms->allocated_memory);
			if (host_cms->allocators[i] == nullptr){
				printf("Not enough space for allocator %d\n", i);
				abort();
			}
		}


		//calculate maximum number of stacks possible
		//and store internally

		//TODO:
		//move table generation to the heap
		//so this can be self contained
		//estimates but table size at .1% of the memory allocated

		uint64_t max_stacks = (bytes_requested-1)/bytes_per_substack + 1;

		//boot hash table here
		poggers::sizing::size_in_num_slots<1> slots_for_table(max_stacks*1.1);
		host_cms->free_table = hash_table::generate_on_device(&slots_for_table, 42);

		//forcing alignment to 16 bytes to guarantee cache alignment for atomics
		my_type * dev_cms = (my_type *) host_cms->allocated_memory->host_malloc_aligned(sizeof(my_type), 16, 0);

		if (dev_cms == nullptr){
			printf("Not enough space for dev ptr\n");
			abort();
		}

		cudaMemcpy(dev_cms, host_cms, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		free(host_cms);

		return dev_cms;


	}

	__host__ static my_type * init(std::size_t bytes_requested){

		return init_backend(bytes_requested, false);

	}

	__host__ static my_type * init_managed(std::size_t bytes_requested){

		return init_backend(bytes_requested, true);
		
	}


	//to free, get handle to memory and just release
	__host__ static void free_cms_allocator(my_type * dev_cms){


		#if COUNTING_CYCLES
		poggers_display_cycles();
		#endif

		my_type * host_cms = (my_type * ) malloc(sizeof(my_type));

		cudaMemcpy(host_cms, dev_cms, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		hash_table::free_on_device(host_cms->free_table);

		heap::free_heap(host_cms->allocated_memory);

		return;

	}


	__device__ allocator * get_randomish_sub_allocator(){

		poggers::hashers::murmurHasher<uint64_t,1> randomish_hasher;

		randomish_hasher.init(42);

		uint64_t very_random_number = clock64();

		return allocators[randomish_hasher.hash(very_random_number) % num_suballocators];

	}


	//malloc splits requests into two groups
	//free list mallocs and stack mallocs
	//we default to stack and only upgrade iff no stack is large enough to handle
	__device__ void * cms_malloc(uint64_t num_bytes){


		#if COUNTING_CYCLES
		uint64_t kernel_start = clock64();
		#endif


		if (allocator::can_malloc(num_bytes)){

			void * val_to_ret = get_randomish_sub_allocator()->template malloc_free_table<hash_table>(num_bytes, free_table, allocated_memory);

			#if COUNTING_CYCLES
			
			uint64_t kernel_end = clock64();

			uint64_t total_time = (kernel_end - kernel_start)/COMPRESS_VALUE;

			atomicAdd((unsigned long long int *) &kernel_counter, (unsigned long long int) total_time);

			atomicAdd((unsigned long long int *) &kernel_traversals, (unsigned long long int) 1);


			#endif


			return val_to_ret;


		} else {

			void * val_to_ret = allocated_memory->malloc(num_bytes);

			#if COUNTING_CYCLES
			
			uint64_t kernel_end = clock64();

			uint64_t total_time = (kernel_end - kernel_start)/COMPRESS_VALUE;

			atomicAdd((unsigned long long int *) &kernel_counter, (unsigned long long int) total_time);

			atomicAdd((unsigned long long int *) &kernel_traversals, (unsigned long long int) 1);


			#endif


			return val_to_ret;

		}

	}


	//force an allocation to fall back to the heap
	//this is for the reporter to stop if from fucking up by creating a new stack
	__device__ void * cms_malloc_force_heap(uint64_t num_bytes){

		return allocated_memory->malloc(num_bytes);
	}

	__device__ void cms_free_force_heap(void * uncasted_address){

		allocated_memory->free_safe(uncasted_address);
	}

	__host__ void * cms_host_malloc(uint64_t num_bytes){



		return host_malloc_wrapper<my_type>(this, num_bytes);

	}

	__host__ void cms_host_free(void * address){

		host_free_wrapper<my_type>(this, address);
	}

	__device__ void cms_free(void * uncasted_address){

		uint64_t home_uint = stack_type::get_home_address_uint(uncasted_address);

		bool found;

		{
			auto tile = free_table->get_my_tile();
			uint16_t temp_val = 0;
			found = free_table->query(tile, home_uint, temp_val);
		}

		if (found){
			stack_type::static_free(uncasted_address);
		} else {
			allocated_memory->free_safe(uncasted_address);
		}

	}


	//local_thread generates and prints report for data struct
	__device__ void one_thread_report(){

		printf("Generating report...\n");

		//force the reporter to live locally
		reporter * my_reporter = (reporter * ) cms_malloc_force_heap(sizeof(reporter));

		my_reporter->init();

		__threadfence();

		allocated_memory->generate_report(my_reporter);

		for (int i =0; i < num_suballocators; i++){

			//printf("%d reporting\n", i);
			allocators[i]->report(my_reporter);
		}

		__threadfence();

		uint64_t malloced = my_reporter->get_stack_bytes_malloced();
		uint64_t free = my_reporter->get_stack_bytes_free();

		double ratio = 1.0;


		if (free == 0){
			ratio = 0;
		} else {
			ratio = ratio*malloced/free;
		}

		



		uint64_t dead_malloced = my_reporter->get_dead_bytes_malloced();
		uint64_t dead_free = my_reporter->get_dead_bytes_free();


		double dead_ratio = 1.0;


		if (dead_free == 0){
			dead_ratio = 0;
		} else {
			dead_ratio = dead_ratio*dead_malloced/dead_free;
		}
		


		uint64_t heap_total = my_reporter->get_heap_bytes_total();

		uint64_t heap_free = my_reporter->get_heap_bytes_free();

		uint64_t heap_used = heap_total - heap_free;

		uint64_t fragmentation_overhead = my_reporter->get_fragmentation();

		//double heap_ratio;

		double heap_ratio = ((double) heap_used)/heap_total;

		double fragmentation = 1.0 - ((double)fragmentation_overhead)/heap_free;


		printf("Live stacks using %llu bytes of %llu bytes given to substacks allocated, %f full\n", malloced, free, ratio);

		printf("Dead stacks using %llu bytes of %llu bytes, %f ratio\n", dead_malloced, dead_free, dead_ratio);

		printf("%llu of %llu stacks report being full\n", my_reporter->get_dead_stacks(), my_reporter->get_total_stacks());

		printf("System Stack utilization: %llu/%llu %f\n", malloced+dead_malloced, free+dead_free, 1.0*(malloced+dead_malloced)/(free+dead_free));

		printf("Heap using %llu/%llu, %f, fragmentation: %f\n", heap_used, heap_total, heap_ratio, fragmentation);

		cms_free_force_heap(my_reporter);

	}

	__host__ void host_report(){
		host_report_wrapper<my_type>(this);
	}

	__host__ static void print_info(){

		printf("Allocator with %llu bytes per substack, %llu sub_allocators, size runs from 4-%llu\n", bytes_per_substack, num_suballocators, 1ULL << (2+maximum_p2));

	}



};


}

}


#endif //GPU_BLOCK_