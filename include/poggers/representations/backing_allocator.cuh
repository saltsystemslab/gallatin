#ifndef SKIPLIST_NODE 
#define SKIPLIST_NODE


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace representations { 

template <typename Data_Storage>
struct skiplist_node<Data_Storage>;

template <typename Data_Storage>
//alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
struct  skiplist_node {

	public:

		Data_Storage local_storage;

		skiplist_node<Data_Storage> * next;
		skiplist_node<Data_Storage> * down;


	__host__ __device__ skiplist_node(){};

	__device__ static skiplist_node<Data_Storage> * request_new_node(skiplist_node<Data_Storage> * node_list, uint64_t max_nodes, uint64_t allocated_count){


		uint64_t allocated_node = atomicAdd((unsigned long long int *) allocated_count, 1ULL);

		if (allocated_node < max_nodes){

			return node_list[allocated_node];

		}

		return NULL;


	}

	__device__ insert_after_me(skiplist_node<Data_Storage> * new_node){

		new_node->next = next;

		next = new_node;

	}

	__device__ insert_new_item(Data_Storage item){

		
	}
		

};



}

}


#endif //GPU_BLOCK_