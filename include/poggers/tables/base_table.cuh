#ifndef BASE_TABLE 
#define BASE_TABLE


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <stdexcept>

#include <cooperative_groups.h>

#include <poggers/tables/recursive_end_table.cuh>

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
template <typename Key, typename Val, template <typename, typename> class Internal_Rep, std::size_t Partition_Size, std::size_t Bucket_Size, template <typename, typename, std::size_t, std::size_t, template <typename, typename> class, std::size_t, template <typename, std::size_t> class , template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class> class Insert_Scheme, std::size_t Max_Probes, template <typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme, template <typename, std::size_t> class Hasher, bool Is_Recursive=false, typename Recursive_Type=poggers::tables::recursive_end_table>
//template<typename Key, typename Val, template <typename, typename> class Internal_Rep, std::size_t Partition_Size,  template <typename, typename, std::size_t, template <typename, typename> class, std::size_t, template <typename, std::size_t> class , template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class> class Insert_Scheme, template <typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme, template <typename, std::size_t> class Hasher, typename Sizing_Type, bool Is_Recursive, typename Recursive_Type>
struct __attribute__ ((__packed__)) static_table {

	using key_type = Key;
	using val_type = Val;


	using insert_scheme_type = Insert_Scheme<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	using my_type = static_table<Key, Val, Internal_Rep, Partition_Size, Bucket_Size, Insert_Scheme, Max_Probes, Probing_Scheme, Hasher, Is_Recursive, Recursive_Type>;


	//tag bits change based on the #of bytes allocated per block
private:



	insert_scheme_type * my_insert_scheme;

	Recursive_Type * secondary_table;

public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	// using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage


	//only allowed to be defined on CPU
	__host__ static_table(){}


	//set the recursive table after initialization
	__host__ void set_recursive_table(Recursive_Type * ext_secondary_table){

		secondary_table = ext_secondary_table;
	}

	__host__ void set_insert_scheme(insert_scheme_type * ext_my_scheme){
		my_insert_scheme = ext_my_scheme;
	}

	template <typename Sizing_Type>
	__host__ static my_type * generate_on_device(Sizing_Type * sizing, uint64_t seed){

		my_type * host_table = (my_type *) malloc(sizeof(my_type));

		uint64_t nslots = sizing->next();

		if (nslots == sizing->end()){
			throw std::runtime_error("Hash Table expects a num_slots from the sizing, but the size container is empty\n");

		}
		host_table->set_insert_scheme(insert_scheme_type::generate_on_device(nslots,seed));

		if (Is_Recursive){

			host_table->secondary_table = Recursive_Type::generate_on_device(sizing, seed*seed);

		} else {

			if (sizing->next() != sizing->end()){
				throw std::runtime_error("Hash Table Does not expect recursion but the size container is not empty.\n");
			}
		}

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_table, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
		free(host_table);

		return dev_version;


	}

	__host__ static void free_on_device(my_type * dev_version){


		my_type host_version;

		cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		insert_scheme_type::free_on_device(host_version.my_insert_scheme);

		if (Is_Recursive){
			Recursive_Type::free_on_device(host_version.secondary_table);
		}

		cudaFree(dev_version);

		return;

	}


	__device__ bool insert(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val){


		// if (Insert_tile.thread_rank() == 0){ printf("Starting Test\n" );}

		if (my_insert_scheme->insert(Insert_tile, key, val)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->insert(Insert_tile, key, val);
			}

			
			return false;

		}

		

	}

	__device__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val){


		// if (Insert_tile.thread_rank() == 0){ printf("Starting Test\n" );}

		if (my_insert_scheme->insert_with_delete(Insert_tile, key, val)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->insert_with_delete(Insert_tile, key, val);
			}

			
			return false;

		}

		

	}

	//inserting from empty does nothing
	__device__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> Insert_tile, Key key){

		Val alt_val = 0;
		return insert_with_delete(Insert_tile, key, alt_val);

	}


	__device__ bool insert_if_not_exists(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val, Val & ext_val, bool & found){


		// if (Insert_tile.thread_rank() == 0){ printf("Starting Test\n" );}

		if (my_insert_scheme->insert_if_not_exists(Insert_tile, key, val, ext_val, found)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->insert_if_not_exists(Insert_tile, key, val, ext_val, found);
			}

			
			return false;

		}

		

	}

	__device__ bool insert_if_not_exists_delete(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val, Val & ext_val, bool & found){


		// if (Insert_tile.thread_rank() == 0){ printf("Starting Test\n" );}

		if (my_insert_scheme->insert_if_not_exists_delete(Insert_tile, key, val, ext_val, found)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->insert_if_not_exists_delete(Insert_tile, key, val, ext_val, found);
			}

			
			return false;

		}

		

	}

	//inserting from empty does nothing
	__device__ bool insert(cg::thread_block_tile<Partition_Size> Insert_tile, Key key){

		Val alt_val = 0;
		return insert(Insert_tile, key, alt_val);

	}


	__device__ bool query(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val & val){

		if (my_insert_scheme->query(Insert_tile, key, val)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->query(Insert_tile, key, val);
			}

			
			return false;

		}

		

	}

	__device__ bool remove(cg::thread_block_tile<Partition_Size> Insert_tile, Key key){

		if (my_insert_scheme->remove(Insert_tile, key)){
			return true;
		} else {

			if (Is_Recursive){
				return secondary_table->remove(Insert_tile, key);
			}

			return false;
		}




	}


	__device__ __inline__ cg::thread_block_tile<Partition_Size> get_my_tile(){

		auto thread_block = cg::this_thread_block();

  	 	cg::thread_block_tile<Partition_Size> insert_tile = cg::tiled_partition<Partition_Size>(thread_block);

  	 	return insert_tile;
 
	}

	__host__ static uint64_t get_block_size(uint64_t nitems_to_insert){
		return 1024;
	}


	__host__ static uint64_t get_num_blocks(uint64_t nitems_to_insert){

		return (Partition_Size*nitems_to_insert-1)/get_block_size(nitems_to_insert) +1;
	}

	__host__ uint64_t host_bytes_in_use(){


		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, this, sizeof(my_type), cudaMemcpyDeviceToHost);

		uint64_t total_bytes = host_version->my_insert_scheme->host_bytes_in_use();

		if (Is_Recursive){
			total_bytes += host_version->secondary_table->host_bytes_in_use();
		}

		cudaFreeHost(host_version);

		return total_bytes;


	}




};

}

}


#endif //GPU_BLOCK_