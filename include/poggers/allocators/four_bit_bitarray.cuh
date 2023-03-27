#ifndef POGGERS_FOUR_BITARRAY
#define POGGERS_FOUR_BITARRAY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>


//STATES of the new scheme
//first bit is locking - always need to lock and unlock for modding
//fully allocated - X000 - no children, can't allocate, no continue
// - This state is identical to both fully allocated nodes and childless nodes
//children available below - X100, children, can't allocate, no continue
//chained alloc - X001, chain extends till X000


namespace cg = cooperative_groups;

//BITMASK Stuff
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define UPPER_BITMASK(index) (BITMASK(64) << index)
#define SET_BIT_MASK(index) ((1ULL << (index)))
#define UNSET_BIT_MASK(index) (~SET_BIT_MASK(index))


#define SET_LOCK_BIT_FOUR(index) ((SET_BIT_MASK(index*4)))
#define SET_CHILD_BIT_FOUR(index) ((SET_BIT_MASK(index*4+1)))
#define SET_ALLOC_BIT_FOUR(index) ((SET_BIT_MASK(index*4+2)))
#define SET_CONTINUE_BIT_FOUR(index) ((SET_BIT_MASK(index*4+3)))


#define UNSET_LOCK_BIT_FOUR(index) (~SET_LOCK_BIT_FOUR(index))
#define UNSET_CHILD_BIT_FOUR(index) (~SET_CHILD_BIT_FOUR(index))
#define UNSET_ALLOC_BIT_FOUR(index) (~SET_ALLOC_BIT_FOUR(index))
#define UNSET_CONTINUE_BIT_FOUR(index) (~SET_CONTINUE_BIT_FOUR(index))


//TODO: assert that nvcc processes this into one bit mask
//intuitively its 
#define READ_ALL_FOUR(index) ((BITMASK(4) << index*4))
//selects 0001 from every quad
#define HAS_LOCK_FOUR_BIT_MASK 0x1111111111111111ULL

//selects 0010 from every quad
#define HAS_CHILD_FOUR_BIT_MASK 0x2222222222222222ULL

//selects 0100 from every quad
#define HAS_ALLOC_FOUR_BIT_MASK 0x4444444444444444ULL

//selects 1000 from every quad
#define HAS_CONTINUE_FOUR_BIT_MASK 0x8888888888888888ULL



//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 





struct four_bit_bitarray {


	uint64_t bits;

	__host__ __device__ four_bit_bitarray(){
		bits = 0ULL;
	}

	__host__ __device__ four_bit_bitarray(uint64_t ext_bits){
		bits = ext_bits;
	}

	__device__ int shrink_index(int index){

		if (index == -1) return index;

		return index/4;
	}

	__device__ int get_random_all_avail(){

		uint64_t tid = threadIdx.x+threadIdx.x;

		int random_cutoff = ((tid*1610612741ULL) % 64);		

		uint64_t both = ((bits) & (bits >> 1));

		both = ((both) & (both >> 1)) & HAS_LOCK_FOUR_BIT_MASK;

		int index = __ffsll(both & UPPER_BITMASK(random_cutoff))-1;

		//option 1 branchless comp?
		//return (index == -1 ? (__ffsll(both) -1) : index);

		//option 2 - think this is truly branchless
		return (index != -1) * index + (index == -1) * (__ffsll(both)-1);

	}

	__device__ int get_first_x_contiguous(uint x){

		uint64_t both = ((bits) & (bits >> 1));

		both = ((both) & (both >> 1)) & HAS_LOCK_FOUR_BIT_MASK;

		//at this point, both is a bitmask of 0001 - if all 4 set
		//need to add this to itself for every p2 of x.
		uint64_t output = ~0ULL;

		while (x != 0){

			int first_bit = __ffsll(x)-1;

			output = (output & (both << first_bit));

			x ^= (1ULL << first_bit);

		}

		return __ffsll(output)-1;

	}


	//lock x contiguous regions simultaneously
	__device__ uint64_t x_lock_mask(int x){


		uint64_t output = 0ULL;

		for (int i=0; i< x; i++){
			output = (output << 4) + 1U;
		}

		return output;

	}


	__device__ int get_random_child(){

		uint64_t tid = threadIdx.x+threadIdx.x;

		int random_cutoff = ((tid*1610612741ULL) % 64);		

		uint64_t both = ((bits) & HAS_CHILD_FOUR_BIT_MASK);

		//both = ((both) & (both >> 1)) & HAS_LOCK_FOUR_BIT_MASK;

		int index = __ffsll(both & UPPER_BITMASK(random_cutoff))-1;

		//option 1 branchless comp?
		//return (index == -1 ? (__ffsll(both) -1) : index);

		//option 2 - think this is truly branchless
		return (index != -1) * index + (index == -1) * (__ffsll(both)-1);

	}

	__device__ int get_first_child(){


		uint64_t both = ((bits) & HAS_CHILD_FOUR_BIT_MASK);

		//both = ((both) & (both >> 1)) & HAS_LOCK_FOUR_BIT_MASK;

		return __ffsll(both)-1;

	}

	__device__ int get_first_child_lock(){

		uint64_t both = ((bits) & (bits >> 1)) & HAS_LOCK_FOUR_BIT_MASK;

		return __ffsll(both)-1;
		
	}

	//Look for valid children, so we want children + ~alloc
	__device__ int get_first_child_only(){

		//condense 0110 into 0010 and select only that bit
		//this reverses the alloc bit - we explicity don't want fully bits
		uint64_t both = ((bits) & (~(bits >> 1))) & HAS_CHILD_FOUR_BIT_MASK;

		return __ffsll(both)-1;
		
	}

	__device__ inline uint64_t reset_all_atomic(uint64_t old, int index){

		return atomicOr((unsigned long long int *) this, READ_ALL_FOUR(index) & old);

	}

	__device__ inline uint64_t set_all_atomic(int index){

				return atomicOr((unsigned long long int *) this, READ_ALL_FOUR(index));

	}

	__device__ inline uint64_t unset_all_atomic(int index){

				return atomicAnd((unsigned long long int *) this, ~READ_ALL_FOUR(index));

	}

	__device__ inline uint64_t unset_bits(uint64_t bits){

		return atomicAnd((unsigned long long int *) this, bits);


	}

	__device__ inline uint64_t set_bits(uint64_t bits){

		return atomicOr((unsigned long long int *) this, bits);


	}

	__device__ inline uint64_t set_lock_bit_atomic(int index){

		return atomicOr((unsigned long long int *) this, SET_LOCK_BIT_FOUR(index));

	}

	__device__ inline uint64_t unset_lock_bit_atomic(int index){

		return atomicAnd((unsigned long long int *) this, UNSET_LOCK_BIT_FOUR(index));

	}


		__device__ inline uint64_t set_child_bit_atomic(int index){

			return atomicOr((unsigned long long int *) this, SET_CHILD_BIT_FOUR(index));

		}

		__device__ inline uint64_t unset_child_bit_atomic(int index){

		return atomicAnd((unsigned long long int *) this, UNSET_CHILD_BIT_FOUR(index));

	}

	__device__ inline uint64_t set_alloc_bit_atomic(int index){

		return atomicOr((unsigned long long int *) this, SET_ALLOC_BIT_FOUR(index));

	}

	__device__ inline uint64_t unset_alloc_bit_atomic(int index){

		return atomicAnd((unsigned long long int *) this, UNSET_ALLOC_BIT_FOUR(index));

	}


	__device__ inline uint64_t set_continue_bit_atomic(int index){

		return atomicOr((unsigned long long int *) this, SET_CONTINUE_BIT_FOUR(index));

	}

	__device__ inline uint64_t unset_continue_bit_atomic(int index){

		return atomicAnd((unsigned long long int *) this, UNSET_CONTINUE_BIT_FOUR(index));

	}

	__host__ __device__ operator uint64_t() const { return bits; }

	__device__ four_bit_bitarray global_load_this(){

		return (four_bit_bitarray) poggers::utils::ldca((uint64_t *) this);

	}

};

}

}


#endif //GPU_BLOCK_