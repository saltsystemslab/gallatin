#ifndef GALLATIN_SINGLE_VECTOR
#define GALLATIN_SINGLE_VECTOR


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>

#include <vector>


namespace gallatin {

namespace data_structs {


	template <typename svec>
	__global__ void svec_init_with_realloc(svec * new_vector, uint64_t nslots){


		uint64_t tid = gallatin::utils::get_tid();

		if (tid != 0) return;

		new_vector->realloc(nslots);


	}

	template <typename svec, typename T>
	__global__ void svec_set_items(svec * new_vector, T * ext_data, uint64_t nslots){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nslots) return;

		new_vector->data[tid] = ext_data[tid];

		new_vector->size = nslots;

	}

	//"single-threaded" vector implementation
	//this is a simple tool to handle map-reduce parallelism
	//in CUDA using Gallatin. - reads, writes, and resizes are handled
	//lazily using one thread - no guarantee of correctness among CG.
	template <typename T>
	struct svector {

		using my_type = svector<T>;

		uint64_t size;
		uint64_t backing_size;
		T * data;

		//temporary pointer for managing incoming pointers




		__device__ svector(uint64_t starting_count){


			uint64_t first_bit_bigger = gallatin::utils::get_first_bit_bigger(starting_count);
			uint64_t rounded_size = (1ULL << first_bit_bigger);

			backing_size = rounded_size;
			size = 0;

			data = (T * ) gallatin::allocators::global_malloc(sizeof(T)*rounded_size);


		}

		//delete memory used by this vector.
		//atm not really threadsafe so only do it if you're sure.
		__device__ void free_vector(){

			gallatin::allocators::global_free(data);

		}


		__device__ void upsize(){

			uint64_t new_backing_size = backing_size*2;

			T * new_data = (T *) gallatin::allocators::global_malloc(sizeof(T)*new_backing_size);

			if (new_data == nullptr){
				asm volatile ("trap;");
			}

			for (uint64_t i = 0; i < size; i++){
				new_data[i] = data[i];
			}

			gallatin::allocators::global_free(data);

			data = new_data;

			backing_size = new_backing_size;

		}

		__device__ void downsize(){

			uint64_t new_backing_size = backing_size/2;


			T * new_data = (T  *) gallatin::allocators::global_malloc(sizeof(T)*new_backing_size);

			if (new_data == nullptr){
				asm volatile ("trap;");
			}

			for (uint64_t i = 0; i < size; i++){
				new_data[i] = data[i];
			}

			gallatin::allocators::global_free(data);

			data = new_data;

			backing_size = new_backing_size;


		}

		__device__ void push_back(T new_item){

			uint64_t my_slot = size;

			size = size+1;

			data[my_slot] = new_item;

			if (size == backing_size) upsize();


		}

		__device__ void insert(T new_item, uint64_t index){

			for (uint64_t i = index+1; i < size; i++){
				data[i-1] = data[i];
			}

			size = size-1;

			if (size == backing_size){
				upsize();
			}

		}

		__device__ void remove(uint64_t index){

			for (uint64_t i = index+1; i < size; i++){
				data[i-1] = data[i];
			}

			size = size-1;

			if (size == backing_size/2 && backing_size != 1){
				downsize();
			}

		}

		__device__ void realloc(uint64_t new_nslots){

			//shortcut if already at correct size.
			if (new_nslots == backing_size) return;

			bool shrinking = (new_nslots < backing_size);

			T * new_data = (T *) gallatin::allocators::global_malloc(sizeof(T)*new_nslots);

			if (new_data == nullptr){
				asm volatile ("trap;");
			}

			uint64_t copy_size = (shrinking)*new_nslots + (!shrinking)*size;

			for (uint64_t i = 0; i < copy_size; i++){
				new_data[i] = data[i];
			}

			if (data != nullptr){
				gallatin::allocators::global_free(data);
			}
			

			data = new_data;

			backing_size = new_nslots;

		}

		__device__ T& operator[](uint64_t index)
		{

		   return data[index];
		   
		}


		//copy host vector to device
		static __host__ my_type * copy_to_device(std::vector<T> external_vector){

			uint64_t size = external_vector.size();

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->data = nullptr;
			host_version->size = 0;
			host_version->backing_size = 0;

			my_type * dev_version = gallatin::utils::move_to_device<my_type>(host_version);

			svec_init_with_realloc<my_type><<<1,1>>>(dev_version, size);

			T * dev_array;

			cudaMalloc((void **)&dev_array, sizeof(T)*size);

			cudaMemcpy(dev_array, external_vector.data(), sizeof(T)*size, cudaMemcpyHostToDevice);

			svec_set_items<my_type, T><<<(size-1)/256+1,256>>>(dev_version, dev_array, size);

			cudaFree(dev_array);

			return dev_version;


		}

		static __host__ my_type * copy_to_device(T * external_array, uint64_t size){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->data = nullptr;
			host_version->size = 0;
			host_version->backing_size = 0;

			my_type * dev_version = gallatin::utils::move_to_device<my_type>(host_version);

			svec_init_with_realloc<my_type><<<1,1>>>(dev_version, size);

			T * dev_array;

			cudaMalloc((void **)&dev_array, sizeof(T)*size);

			cudaMemcpy(dev_array, external_array, sizeof(T)*size, cudaMemcpyHostToDevice);

			svec_set_items<my_type, T><<<(size-1)/256+1,256>>>(dev_version, dev_array, size);

			cudaFree(dev_array);

			return dev_version;


		}


		//static __host__ my_type * move_to_device

	};


}


}


#endif //end of queue name guard