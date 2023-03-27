#ifndef END_TABLE 
#define END_TABLE


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <stdexcept>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace tables {


//The insert schemes are in charge of device side inserts
//they handle memory allocations for the host-side table
// and most of the logic involved in inserting/querying



//given a 


//probing schemes map keys to buckets/slots in some predefined pattern


// template <typename Key, typename Val, std::size_t Partition_Size, template <typename, typename> class Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme>


// typename Key
// typename Val
// typename <typename, typename> class Internal_Rep
// std::size_t Partition_Size

// template <typename, typename, std::size_t, template <typename, typename> class, std::size_t, template <typename, std::size_t> class , template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class> Insert_Scheme
// <typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme
// template <typename, std::size_t> class Hasher
// typename Sizing_Type
// bool Is_Recursive
// typename Recursive_type;
//emplate <typename, typename> class Internal_Rep, std::size_t Partition_Size
struct __attribute__ ((__packed__)) recursive_end_table {


	//tag bits change based on the #of bytes allocated per block


public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	// using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	 using my_type = recursive_end_table;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage


	//only allowed to be defined on CPU
	__host__ recursive_end_table(){}


	// //set the recursive table after initialization
	// __host__ void set_recursive_table(Recursive_Type * ext_secondary_table){

	// 	secondary_table = ext_secondary_table;
	// }

	// __host__ void set_insert_scheme(,y* ext_my_scheme){
	// 	my_insert_scheme = ext_my_scheme;
	// }

	template< typename Sizing_type>
	__host__ static my_type * generate_on_device(Sizing_type * sizing, uint64_t seed){


		//this should never be called
		//a formality so that recursive calls have something to look at
		assert (sizing->next() == sizing->end());

		//obviously never call this
		my_type * dev_version = NULL;

		// cudaMalloc((void **)&dev_version, sizeof(my_type));

		// cudaMemcpy(dev_version, &host_table, sizeof(my_type), cudaMemcpyHostToDevice);

		return dev_version;



	}


	__host__ static void free_on_device(my_type * dev_version){

		return;

	}

	//dummy templates to allow the end table to interface with any table type - any requests to the end table fail.
	template <class... Us>
	__device__ bool insert(Us... pargs){
		return false;
	}
	template <class... Us>
	__device__ bool insert_if_not_exists(Us... pargs){
		return false;
	}

	template <class... Us>
	__device__ bool insert_if_not_exists_delete(Us... pargs){
		return false;
	}

	template <class... Us>
	__device__ bool query(Us... pargs){
		return false;
	}

	template <class... Us>
	__device__ bool remove(Us... pargs){
		return false;
	}

	__host__ uint64_t host_bytes_in_use(){


		return 0;

	}

	template <class... Us>
	__device__ bool insert_with_delete(Us... pargs){
		return false;
	}

	template <class... Us>
	__host__ uint64_t get_fill(Us... pargs){

		return 0;
	}

	template <class... Us>
	__host__ uint64_t get_num_slots(Us... pargs){

		return 0;
	}





};

}

}


#endif //GPU_BLOCK_