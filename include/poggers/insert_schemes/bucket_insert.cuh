#ifndef BUCKET_INSERT 
#define BUCKET_INSERT


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
struct __attribute__ ((__packed__)) bucket_insert {


	//tag bits change based on the #of bytes allocated per block
private:



	Internal_Rep<Key, Val> * slots;

	const uint64_t num_buckets;
	const uint64_t seed;



public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	using my_type = bucket_insert<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	using rep_type = Internal_Rep<Key, Val>;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ bucket_insert(): num_buckets(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ bucket_insert(Internal_Rep<Key, Val> * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): num_buckets(ext_nslots), seed(ext_seed){
		
		slots = ext_slots;
	}


	__host__ static my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

		Internal_Rep<Key, Val> * ext_slots;

		uint64_t min_buckets = (ext_nslots-1)/Bucket_Size+1;

		uint64_t true_nslots = min_buckets*Bucket_Size;


		cudaMalloc((void **)& ext_slots, true_nslots*sizeof(Internal_Rep<Key, Val>));
		cudaMemset(ext_slots, 0, true_nslots*sizeof(Internal_Rep<Key, Val>));

		my_type host_version (ext_slots, min_buckets, ext_seed);

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

       		insert_slot = insert_slot % num_buckets;

       		insert_slot = insert_slot*Bucket_Size;


       		for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+=Partition_Size){

       			uint64_t my_insert_slot = insert_slot+i;
    

       	

	       		//printf("checking_for_slot\n");

	       		bool ballot = false;

	       		if (slots[my_insert_slot].is_empty()){
	       			ballot = true;
	       		}

       			auto ballot_result = insert_tile.ballot(ballot);

       		


	       		while (ballot_result){

	       			const auto leader = __ffs(ballot_result)-1;

	       			if (leader == insert_tile.thread_rank()){
	       				if (slots[my_insert_slot].atomic_swap(key, val)){
	       					insert_tile.ballot(true);

	       					return true;
	       				} 

	       				
	       				//on failure, ballot
	       				insert_tile.ballot(false);

	       			



	       			} else {

	       				//if leader succeeds return
	       				if (insert_tile.ballot(false)){
	       					return true;
	       				}
	       			}

	       			//if we made it here no successes, decrement leader
	       			ballot_result  ^= 1UL << leader;

	       		}


	       		}

	       	}



	     	return false;

	}

	__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key key){

		//first step is to init probing scheme



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       		insert_slot = insert_slot % num_buckets;

       		insert_slot = insert_slot*Bucket_Size;

       		bool bucket_empty = false;


       		for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+=Partition_Size){

       			uint64_t my_insert_slot = insert_slot+i;
    

       	

	       		//printf("checking_for_slot\n");

	       		bool ballot = false;

	       		if (slots[my_insert_slot].contains(key)){
	       			ballot = true;
	       		} else if (slots[my_insert_slot].is_empty()){
	       			bucket_empty = true;
	       		}


       			auto ballot_result = insert_tile.ballot(ballot);

       		


	       		while (ballot_result){

	       			const auto leader = __ffs(ballot_result)-1;

	       			if (leader == insert_tile.thread_rank()){
	       				if (slots[my_insert_slot].atomic_reset(key)){
	       					insert_tile.ballot(true);

	       					return true;
	       				} 

	       				
	       				//on failure, ballot
	       				insert_tile.ballot(false);

	       			



	       			} else {

	       				//if leader succeeds return
	       				if (insert_tile.ballot(false)){
	       					return true;
	       				}
	       			}

	       			//if we made it here no successes, decrement leader
	       			ballot_result  ^= 1UL << leader;

	       		}


	       		}


	       		//ballot here on empty

	       		if (insert_tile.ballot(bucket_empty)){
	       			return false;
	       		}

	       	}



	     	return false;

	}

	__device__ __inline__ bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val& ext_val){

		//first step is to init probing scheme



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       		insert_slot = insert_slot % num_buckets;

       		insert_slot = insert_slot*Bucket_Size; // + insert_tile.thread_rank();

       		//printf("checking_for_slot\n");

       		for (int i = insert_tile.thread_rank(); i < Bucket_Size; i+=Partition_Size){



     			uint64_t my_insert_slot = insert_slot+i;


	       		bool ballot = false;

	       		if (slots[my_insert_slot].contains(key)){
	       			ext_val = slots[my_insert_slot].get_val(key);
	       			ballot = true;
	       		}

       			auto ballot_result = insert_tile.ballot(ballot);

	       		if (ballot_result){

	       			ext_val = insert_tile.shfl(ext_val, __ffs(ballot_result)-1);
	       			return true;


	       		}

      	 		//if true we end here
       			//an empty slot was found, so we could have inserted here

	       		if (slots[my_insert_slot].is_empty()){
	       			ballot = true;
	       		}


	       		ballot_result = insert_tile.ballot(ballot);

	       		if (ballot_result) return false;

	       
	      	}


	      }
     	

	     return false;
     	

		}



};

}

}


#endif //GPU_BLOCK_