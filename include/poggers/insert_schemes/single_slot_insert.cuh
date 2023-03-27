#ifndef SINGLE_SLOT_INSERT 
#define SINGLE_SLOT_INSERT


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


template <typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size, template <typename, typename> class Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme>

//template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_Probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_Probes>
struct __attribute__ ((__packed__)) single_slot_insert {


	//tag bits change based on the #of bytes allocated per block

	static_assert(Partition_Size == Bucket_Size, "For Single Slot Insert bucket size and partition_size must be equal. (Just go use bucket_insert!)\n");

private:



	Internal_Rep<Key, Val> * slots;
	const uint64_t nslots;
	const uint64_t seed;



public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	using my_type = single_slot_insert<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	using rep_type = Internal_Rep<Key, Val>;


	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ single_slot_insert(): nslots(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ single_slot_insert(Internal_Rep<Key, Val> * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): nslots(ext_nslots), seed(ext_seed){
		
		slots = ext_slots;
	}


	__host__ static my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

		Internal_Rep<Key, Val> * ext_slots;

		cudaMalloc((void **)& ext_slots, ext_nslots*sizeof(Internal_Rep<Key, Val>));
		cudaMemset(ext_slots, 0, ext_nslots*sizeof(Internal_Rep<Key, Val>));

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

	__device__ __inline__ bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		//first step is to init probing scheme



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

			if (insert_tile.thread_rank() == 0){
       		insert_slot = insert_slot % nslots;

       		//printf("checking_for_slot\n");

       		if (slots[insert_slot].is_empty()){

       			//printf("found to be empty\n");

       			if (slots[insert_slot].atomic_swap(key, val)){
       				insert_tile.ballot(true);

       				//printf("Inserted\n");
       				return true;
       				}
       			}

       			insert_tile.ballot(false);

      	 	} else {

      	 		bool found = insert_tile.ballot(false);
      	 		if (found) return true;
      	 	}


     	}


     	return false;

	}

	__device__ __inline__ bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val& ext_val){

		//first step is to init probing scheme



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

			if (insert_tile.thread_rank() == 0){
       		insert_slot = insert_slot % nslots;

       		//printf("checking_for_slot\n");

       		if (slots[insert_slot].contains(key)){

       			ext_val = slots[insert_slot].get_val(key);

       			insert_tile.ballot(true);

       			ext_val = insert_tile.shfl(ext_val, 0);

       			return true;

       		} else {
       			insert_tile.ballot(false);
       		}

       		if (slots[insert_slot].is_empty()){

       			insert_tile.ballot(true);

	      		return false;
      	 	} else {

      	 		insert_tile.ballot(false);
      	 	}


     	} else {

     		bool found = insert_tile.ballot(false);
     		if (found) {
     			ext_val = insert_tile.shfl(ext_val, 0);
     			return true;
     		}

     		bool empty = insert_tile.ballot(false);

     		if (empty) return false;

     	}


     	

		}

		return false;

	}



};

}

}


#endif //GPU_BLOCK_