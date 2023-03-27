#ifndef DUMMY_SCHEME 
#define DUMMY_SCHEME


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace insert_schemes {


//The insert schemes are in charge of device side inserts
//they handle memory allocations for the host-side table
// and most of the logic involved in inserting/querying



//given a 


//probing schemes map keys to buckets/slots in some predefined pattern
//template <typename Key, std::size_t Partition_Size, typename Internal_Rep, template <typename, std::size_t> class Hasher, template <typename, std::size_t, template <typename, std::size_t > class, std::size_t > class Probing_Scheme, std::size_t Max_Probes >


template <typename Key, std::size_t Partition_Size, typename Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme>

//template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_Probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_Probes>
struct __attribute__ ((__packed__)) dummy_scheme {


	//tag bits change based on the #of bytes allocated per block
private:



	Internal_Rep * slots;
	const uint64_t nslots;
	const uint64_t seed;



public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	//using probing_scheme_type = Probing_Scheme<Key, Partition_Size, Hasher, Max_Probes>;
	//using my_type = dummy_scheme<Key, Partition_Size, Internal_Rep, Hasher, Probing_Scheme, Max_Probes>;
	using my_type = dummy_scheme<Key, Partition_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ dummy_scheme() : nslots(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ dummy_scheme(Internal_Rep * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): nslots(ext_nslots), seed(ext_seed){
		
		slots = ext_slots;
	}


	__host__ static my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

		Internal_Rep * ext_slots;

		cudaMalloc((void **)& ext_slots, ext_nslots*sizeof(Internal_Rep));
		cudaMemset(ext_slots, 0, ext_nslots*sizeof(Internal_Rep));

		my_type host_version (ext_slots, ext_nslots, ext_seed);

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		return dev_version;



	}

	__host__ static void free_on_device(my_type * dev_version){


		my_type host_version;

		cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaFree(host_version.slots);

		cudaFree(dev_version);

		return;

	}

	//no need for explicit destructor struct hash no memory components

};

}

}


#endif //GPU_BLOCK_