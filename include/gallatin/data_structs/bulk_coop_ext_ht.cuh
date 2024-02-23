#ifndef GALLATIN_BULK_COOP_HT
#define GALLATIN_BULK_COOP_HT


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/data_structs/fixed_vector.cuh>


#define KEY_IS_HASH 0

namespace gallatin {

namespace data_structs {

	//Pipeline

	//insert
	// - alloc new node
	// - set next of node to current


	template<typename ht>
	__global__ void init_vectors(ht * table , uint64_t n_threads){
		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= n_threads) return;



		table->init_vector(tid);

	}

	template<typename ht, typename Key , typename Val>
	__global__ void bulk_insert_kernel(ht * table, Key * keys, Val * vals, uint64_t nitems){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nitems) return;

		table->device_bulk_insert(keys[tid], vals[tid]);

	}

	template <typename Key, typename Val>
	struct extendible_key_val_pair {

		Key key;
		Val val;

	};

	template <typename Key, typename Val, uint64_t n_vectors>
	struct bulk_ext_table{

		using my_type = bulk_ext_table<Key, Val, n_vectors>;

		using key_val_type = extendible_key_val_pair<Key, Val>;


		using fixed_vector_type = gallatin::data_structs::fixed_vector<key_val_type, 32, 8192>;



		fixed_vector_type directory[n_vectors];


		static __host__ my_type * generate_on_device(){



			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);

			init_vectors<my_type><<<(n_vectors-1)/256+1,256>>>(device_version, n_vectors);

			return device_version;

		}

		static __host__ void free_on_device(my_type * dev_version){


			printf("Free is currently not implemented\n");
			return;
		}

		__device__ uint64_t get_hash(Key key){

			//todo seed
			return gallatin::hashers::MurmurHash64A(&key, sizeof(Key), 42) % n_vectors;
		}



		__device__ void init_vector(uint64_t id){

			directory[id].init();
			directory[id].add_new_backing(0);
			//directory[id].add_new_backing(1);
			// directory[id].add_new_backing(2);

		}


		__device__ bool insert(Key ext_key, Val ext_val){

			key_val_type packed_pair{ext_key, ext_val};


			#if KEY_IS_HASH
			uint64_t hash = ext_key % n_vectors;
			#else
			uint64_t hash = get_hash(ext_key);
			#endif

			return (directory[hash].insert(packed_pair) != ~0ULL);


		}


		__device__ bool device_bulk_insert(Key ext_key, Val ext_val){

			key_val_type packed_pair{ext_key, ext_val};


			#if KEY_IS_HASH
			uint64_t hash = ext_key % n_vectors;
			#else
			uint64_t hash = get_hash(ext_key);
			#endif

			cg::coalesced_group full_warp_team = cg::coalesced_threads();

   		 	cg::coalesced_group coalesced_team = labeled_partition(full_warp_team, hash);

			return (directory[hash].bulk_insert(coalesced_team, packed_pair) != ~0ULL);


		}





		__host__ void bulk_insert(Key * keys, Val * vals, uint64_t nitems){


			//sort
			thrust::sort_by_key(thrust::device, keys, keys+nitems, vals);
			bulk_insert_kernel<my_type, Key, Val><<<(nitems-1)/256+1, 256>>>(this, keys, vals, nitems);


		}

	};



}


}


#endif //end of queue name guard