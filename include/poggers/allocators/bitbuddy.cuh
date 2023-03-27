#ifndef POGGERS_BITBUDDY
#define POGGERS_BITBUDDY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;


#define LEVEL_CUTOF 0

#define PROG_CUTOFF 3


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 




__global__ void setup_first_level(uint64_t * items, uint64_t num_uints_lowest_level, uint64_t num_items_lowest_level){

	//float through each level
	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_uints_lowest_level) return;

	uint64_t my_uint = 0ULL;

	for (int i = 0; i < 32; i++){

		if (tid * 32 + i < num_items_lowest_level){
			my_uint |= (3ULL << (2*i));
		}

	}

	items[tid] = my_uint;

}

//Given a next level, initialize the level above it. This is (AFAIK) fanout agnostic.


//setup needs to check how many items there are

__global__ void setup_level(uint64_t * current_level, uint64_t num_items_in_level, bool clean){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_items_in_level) return;


	//for 31
	uint64_t my_uint64_t = tid/32;

	uint64_t my_inner = tid % 32;

	if (!clean && tid == (num_items_in_level-1)){

		//set to 10
		atomicOr((unsigned long long int *) current_level + my_uint64_t, (1ULL << (my_inner*2)));

	} else {
		atomicOr((unsigned long long int *) current_level + my_uint64_t, (3ULL << (my_inner*2)));
	}
	

}

//the bitbuddy allocator (patent pending)
//uses teams of buddies managed with bitmaps to quickly allocate and deallocate large blocks of memory
//this allocator assigns two bits to every allocation type: one bit for has_valid_children and one bit for allocable
//Together these allow for fast traversal of the tree while still maintaining constant-time allocs once a suitable match has been found.
//The process is this: The size at the top is known, along with a target
//While 


//TODO: Ask prashant about 3rd bit per item! This could handle allocations of scaling size. - Since fanout is 32x, we can build larger-ish allocations by grabbing contiguous segments
//and then mark that those segments are together with a unary counter.


//TODO list:
// 1) Init - every item should contribute bits to layer above it, repeat until only one layer is left
// 2) Malloc - taken from top rec

//Bit ordering
// Children - available  / fully available
//0x3 for full children allocs otherwise 0x2.
//we don't have to worry about other cases on boot.


//bit configurations
// 00 - All children allocated
// 11 - All children free
// 01 - Main item is alloced
// 10 - Some children are alloced


//New ordering - simplifies some ops
// 11 - all available
// 10 - children available
// 00 - alloced as whole node
// 01 - fully alloced

//swap procedure - setting the first bit to 0 means that the state of the system is not configurable by other threads
// this is becuase both alloced as whole and fully alloced are end states.
// so when allocing a node we swap out the first bit to 0 and observe the state
// if it was already 0, we didn't do anything and do not own this node (read failure)
// if it was 1, we might own the node! check if previous state was 10
// if that is the case, roll back to 10... Whoops
// otherwise the node has been successfully alloced

//When rolling up the list we unset the other bit
//if the observed state was 10 or 11, we're done and are good
//if the state was 00 or 01 before we did something wrong and did not really allocate
//this necessitates a rollback of our changes in the lower levels
//and a reset to the original 01 if that was the previous state.

//big change - when allocating a new node, we should originally swap to 01 and disable the node
//then float up
//this allows for 00 nodes to not exist unless the node is explicitly fully alloced
//this maintains that the top 00 found in an items path is the item on free
//so we start at the top and float down until we see a 00. 


//updated design: set the lock bit before every op
// mapping: lock bit / control bit
// all avail: 1 / 1
// float child: 0 / 1
// occupado: 0 / 0

struct bitbuddy_allocator {

	uint64_t num_levels;
	//uint64_t top_size;

	void * memory;

	uint64_t_bitarr ** levels;

	uint64_t bytes_at_top;



	//every level compresses 32 items
	//potential overestimation here
	__host__ __device__ static uint64_t get_num_items_next_level(uint64_t level_below){


		return (level_below-1)/32+1;



	}


	__host__ __device__ static uint64_t get_size_next_level(uint64_t level_below){
		return level_below*32;
	}

	//Only called on the bytes_per_level, so always save.
	__host__ __device__ static uint64_t get_size_level_below(uint64_t level){

		return level/32;

	}

	//compress index into 32 bit index
	__host__ __device__ static int shrink_index(int index){
		if (index == -1) return index;

		return index/2;
	}



	static __host__ bitbuddy_allocator * generate_on_device(void * ext_memory, uint64_t num_bytes, uint64_t bytes_at_lowest_level){


		//uint64_t num_items_lowest_level = (num_bytes)/bytes_at_lowest_level

		//this should correct for it.
		uint64_t num_items_current_level = (num_bytes)/bytes_at_lowest_level;

		//this is bytes per item
		uint64_t num_bytes_current_level = bytes_at_lowest_level;

		//With the current scheme, we can pack up to 32 items into one bin simultaneously.
		//uint64_t num_uints_lowest_level = (num_items_lowest_level -1 )/32 + 1;

		//I think setup is identical regardless of fanout? that's pretty neat. May require special control logic

		std::vector<uint64_t * > ext_levels;

		uint64_t * current_level;

		uint64_t num_levels = 0;

		bool clean = true;


		while (num_items_current_level > 1){

			//the number of items in the next level is also the number of uint64s needed to contain the current level!
			uint64_t num_items_next_level = get_num_items_next_level(num_items_current_level);

			uint64_t * current_level_bitvector;

			cudaMalloc((void ** )&current_level_bitvector, sizeof(uint64_t)*num_items_next_level);

			cudaMemset(current_level_bitvector, 0, sizeof(uint64_t)*num_items_next_level);

			//this call will init the layer.
			setup_level<<<(num_items_current_level-1/512+1), 512>>>(current_level_bitvector, num_items_current_level, clean);

			ext_levels.push_back(current_level_bitvector);


			//push bytes up*32 to keep track
			

			//need to account for offsets if the current level is not perfect
			//this floats up
			if (num_items_current_level % 32 != 0){
				clean = false;
			}

			//last level don't iterate

			if (num_items_next_level > 1){
				num_bytes_current_level = get_size_next_level(num_bytes_current_level);
			}

			num_items_current_level = num_items_next_level;

			num_levels+=1;




		}


		//at the end, num_items current_level = 1, final level is malloced. //num levels is correct, and the bitvectors are initialized.
		//I think this is all correct, will assert in boot tests

		uint64_t ** levels_arr;

		cudaMalloc((void **)&levels_arr, num_levels*sizeof(uint64_t *));

		cudaMemset(levels_arr, 0, sizeof(uint64_t *)*num_levels);


		std::vector<uint64_t * > ext_levels_rev;

		//Aren't c++ structs wonderful? loop through arrays in reverse.
		for (auto it = ext_levels.rbegin(); it != ext_levels.rend(); ++it){

			ext_levels_rev.push_back(*it);

		}


		cudaMemcpy(levels_arr, ext_levels_rev.data(), sizeof(uint64_t *)*num_levels, cudaMemcpyHostToDevice);

		//Todo: clean this up, no need for two counters
		assert(ext_levels.size() == num_levels);
		
		//now that we have all of this, construct the host version

		bitbuddy_allocator * host_version;

		cudaMallocHost((void **)&host_version, sizeof(bitbuddy_allocator));

		host_version->num_levels = ext_levels.size();

		host_version->levels = (uint64_t_bitarr **)  levels_arr;

		host_version->memory = ext_memory;

		host_version->bytes_at_top = num_bytes_current_level;

		bitbuddy_allocator * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(bitbuddy_allocator));

		cudaMemcpy(dev_version, host_version, sizeof(bitbuddy_allocator), cudaMemcpyHostToDevice);

		cudaFreeHost(host_version);

		return dev_version;

	}




	static __host__ void  free_on_device(bitbuddy_allocator * dev_allocator){


		bitbuddy_allocator host_alloc;

		cudaMemcpy(&host_alloc, dev_allocator, sizeof(bitbuddy_allocator), cudaMemcpyDeviceToHost);

		uint64_t ** host_array;

		cudaMallocHost((void **)&host_array, sizeof(uint64_t *)*host_alloc.num_levels);

		cudaMemcpy(host_array, host_alloc.levels, sizeof(uint64_t *)*host_alloc.num_levels, cudaMemcpyDeviceToHost);


		for (int i =0; i < host_alloc.num_levels; i++){

			cudaFree(host_array[i]);

		}

		cudaFree(host_alloc.levels);

		cudaFree(dev_allocator);

		cudaFreeHost(host_array);


	}


	//Malloc/Free helpers
	//return true if at level
	__device__ bool check_at_level(int & level, uint64_t bytes_at_level, uint64_t bytes_needed){

		//if descended to the lowest level you can't go deeper
		if (level >= (num_levels-1)){

			return true;

		} else if (bytes_at_level <= bytes_needed && get_size_next_level(bytes_at_level) > bytes_needed){
			return true;
		}

		return false;

	}

	//check if it is safe to move items below us
	__device__ bool inline can_descend(int & level){

		return level < (num_levels-1);
	}

	__device__ bool inline descend(int & level, uint64_t & offset){

		int index;
		if (level >= LEVEL_CUTOF){

			index = shrink_index(levels[level][offset].get_random_active_bit_control());

		} else {

			index = shrink_index(levels[level][offset].get_first_active_bit_control());

		}

		if (index == -1) return false;

			offset = offset*32+index;

			level += 1;

			return true;


		}

		__device__ bool inline descend_to_bit(int & level, uint64_t & offset, int & index){


			offset = offset*32+index;
			level+=1;
			return true;

		}


	__device__ bool inline ascend(int & level, uint64_t & offset){

		if (level == 0) return false;

		level -=1;

		offset = offset/32;

		return true;

	}



	//opcodes
	//0 - failure - abort and redo
	//1 - regular success - continue
	//2 - match into regular
	__device__ int float_up_level(int level, uint64_t & offset, uint64_t & prev_level){

		//this gets 3 uses so we'll use up a register to accelerate
		int index = offset % 32;

		uint64_t internal_offset = offset/32;

		if (prev_level != 0ULL){

			//convert to 00
			//assume 01

			prev_level = levels[level][internal_offset].unset_lock_bit_atomic(index);


			//previous state was unlocked, this is good
			if (prev_level & SET_FIRST_BIT(index)){

				//update prev level for float up
				prev_level = prev_level | SET_FIRST_BIT(index);

				return 1;
			}


			//state is 01 - we have met a correct route
			if (prev_level & SET_SECOND_BIT(index)){
				return 2;
			}

			//state is 00 - abort
			return 0;


		} else {

			//prev is 00'd out - so we can unset this bit
			//by definition it must be set to 01

			prev_level = levels[level][internal_offset].unset_lock_bit_atomic(index);

			//do this for consistency
			if ((prev_level & SET_FIRST_BIT(index))){

				printf("Bug in float up.\n");

				//assert(!(prev_level & SET_FIRST_BIT(index)));
			}

			prev_level = levels[level][internal_offset].unset_control_bit_atomic(index);

			assert(prev_level & SET_SECOND_BIT(index));


			prev_level = prev_level &  ~SET_SECOND_BIT(index);


			return 1;
			
		}

	}


	//In float up, we are going to transition 11 to 01 keys
	//10 or 00 trigger an abort and rollback of the changes, starting from the bottom.
	__device__ bool float_up(int lowest_level, uint64_t lowest_offset){


		uint64_t offset = lowest_offset;

		int level = lowest_level-1;

		if (level < 0) return;

		uint64_t items = levels[lowest_level][offset];

		do {


				//printf("Thread %llu starting with %llu:  %llu\n", threadIdx.x, offset, items);
				int result = float_up_level(level, offset, items);

				//printf("Thread %llu float level with result %d\n", threadIdx.x, result);


				if (result == 0) return false;
				if (result == 2) return true;

				//items is updated in prev layer
				//items = levels[level][offset];


		} while (ascend(level, offset));

		return true;



	}

	__device__ void * offset_to_malloc(int & level, uint64_t & offset, int & index){




		//correction for level +1 - remove if segfault lol
		uint64_t bytes_at_level = move_size_down(bytes_at_top, level);


		return (void *) ((uint64_t ) memory + (offset*32+index)*bytes_at_level);



	}

	//malloc will swap a 11 to a 00
	//so set first bit.
	//request an allocation from this level

	//valid states are 11, 01, 00
	//so unset is fine


	inline __device__ bool assert_correct_setup(void * allocation){

		int level = num_levels-1;

		uint64_t offset = cast_to_offset(allocation);

		uint64_t clipped_offest = offset/32;

		int index = offset % 32;

		uint64_t items = levels[level][clipped_offest];


		while(ascend(level, offset)){

			clipped_offest = offset/32;

			index = offset % 32;

			//items is all 0
			if (!items){

				//should be 00
				if (levels[level][clipped_offest] & READ_BOTH(index)){

					printf("Error when asserting correct for 00.\n");

				} else {
					items = levels[level][clipped_offest];
					continue;
				}

			}

			//items is all 11
			if (!~items){

				if (__popcll(levels[level][clipped_offest] & READ_BOTH(index)) == 2){

					items = levels[level][clipped_offest];
					continue;
				} else {

					printf("Error when asserting correct for 11.\n");
				}

			}

			if ((levels[level][clipped_offest] & READ_BOTH(index)) == (1ULL << index)){

				items = levels[level][clipped_offest];
				continue;

			} else {

				printf("Error when asserting correct for 01.\n");
			}



		}

	}

	inline __device__ void * malloc_from_level(int level, uint64_t offset){
	//inline __device__ void * malloc_from_level(int & level, uint64_t & offset, int & index){


		//keep a local copy?

		//to pull from the level we gotta have all the bits
		int index = shrink_index(levels[level][offset].get_random_active_bit_full());


		while (index != -1){

			uint64_t old_bitarr = levels[level][offset].unset_both_atomic(index);

			if (__popcll(old_bitarr & READ_BOTH(index)) == 2){


				//success! was a 11, set to 00
				//levels[level][offset].unset_both_atomic();

				return offset_to_malloc(level, offset, index);



				


			}

			//otherwise, undo
			//if 01 return to 01
			//else was 00 - unchanged
			if (old_bitarr & SET_SECOND_BIT(index)){

				levels[level][offset].set_control_bit_atomic(index);

			}

			//force reload to detect changes.
			levels[level][offset].global_load_this();
			index = shrink_index(levels[level][offset].get_random_active_bit_full());


		}

		//failed to malloc at this level
		return nullptr;

	}


	__device__ void * malloc(uint64_t num_bytes){


		int num_rounds = 0;

		while (num_rounds < PROG_CUTOFF){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			void * result = malloc_check(num_bytes);

			active_threads.sync();


			if (result != nullptr){
				return result;
			} 

			num_rounds += 1;

			//printf("progressing in main malloc\n");

		}


		return nullptr;


	}



	__device__ int select_random_full_index(int level, uint64_t offset){

		while (true){

			int index = shrink_index(levels[level][offset].get_random_active_bit_full());



			if (index == -1){
				correct_up_v3(level, offset, levels[level][offset], 0);
				return -1;
			}

			uint64_t old = levels[level][offset].unset_both_atomic(index);

			if ((old & ~READ_BOTH(index)) == 0ULL){

				correct_up_v3(level, offset, old, index);
				//correct_up_v3(level, offset, (old & ~READ_BOTH(index)));
			}

			if (__popcll(old & READ_BOTH(index)) == 2){
				return index;
			}


			levels[level][offset].reset_both_atomic(old, index);


		}


	}


	__device__ int select_and_set_random_index(int level, uint64_t offset){


		while (true){

		int index;

		if (level < LEVEL_CUTOF){

			index = shrink_index(levels[level][offset].get_random_active_bit_control_only());

			if (index == -1) index = shrink_index(levels[level][offset].get_random_active_bit_full());


		} else {

			index = shrink_index(levels[level][offset].get_random_active_bit_control());

		}


		if (index == -1){
			return -1;
		}


		levels[level][offset].unset_lock_bit_atomic(index);

		if (levels[level][offset] & SET_SECOND_BIT(index)){

			return index;

		}


		}

	}


	__device__ void * malloc_check(uint64_t num_bytes){




		int level = 0;

		uint64_t offset = 0;

		int index;

		uint64_t bytes_at_level = move_size_down(bytes_at_top, level);

		while (true){


			//printf("Bug in this loop?\n");


			//optional load
			uint64_t current_level = levels[level][offset].global_load_this();

			//printf("In this level: %d/%llu %llu: %llx\n", level, num_levels, offset, current_level);


			if (!check_at_level(level, bytes_at_level, num_bytes)){


			index = select_and_set_random_index(level, offset);

			//fail as no path from this level.
			if (index == -1){

				//mayve enable>
				//correct_up_v2(level, offset);


				if (ascend(level, offset)){
					continue;
				} else {
					return nullptr;
				}
			}


			offset = offset*32+index;
			level+=1;

			} else {

				//at level
				index = select_random_full_index(level, offset);


				if (index == -1){
					//maybe enable?
					//correct_up_v2(level, offset);
					ascend(level, offset);
					continue;

				}




				void * alloc  = offset_to_malloc(level, offset, index);



				//uint64_t items = levels[level][offset];


				return alloc;


			}


		}




	}


	//Basic procedures -
	//while there are routes to take, descend down.
	//if there are no routes, ascend up and select a new bit (read old atomically?)
	//when the level we are at is satisfactory for a malloc,
	__device__ void * malloc_check_old(uint64_t num_bytes, bool & success){

		int level = 0;

		uint64_t offset = 0;

		uint64_t bytes_at_level = bytes_at_top;


		int progress = 0;

		//until entirely full, we should always try to malloc.
		while (progress < PROG_CUTOFF){


			//this safeguards falling too low.
			if (check_at_level(level, bytes_at_level, num_bytes)){

				//we are at the correct level
				//let's pick an index to allocate from

				

				void * alloc = malloc_from_level(level, offset);
				//if this is true, one of the items at this level is now 

				if (alloc != nullptr){

					//printf("Thread %llu did alloc\n", threadIdx.x);

					if (float_up(level, offset)){


						//printf("Thread %llu has exited with alloc\n", threadIdx.x);

						success = true;

						return alloc;
					} else {

						//printf("Thread %llu failed to float\n", threadIdx.x);
						//we hit an incorrect state, revert
						free(alloc);

					}


				}


				if (!ascend(level, offset)){
					return nullptr;
				}

				//printf("Thread %llu reaches\n", threadIdx.x);

				progress+=1;

				levels[level][offset].global_load_this();


			} else {





			if (!descend(level, offset)){

					if (!ascend(level, offset)){
						//can not malloc?
						return nullptr;
					}

					//repull new level from memory.
					//some changes must have occured, or a lower level would have been available.
					levels[level][offset].global_load_this();

			}




			}

			


		}


		success = false;
		return nullptr;

	}


	__device__ void correct_up_v2(int level, uint64_t offset, uint64_t items){

		if (level == 0) return;

		do {


			//uint64_t items = levels[level][offset].global_load_this();

			//if all bits unset
			if ((!items) && (level > 0)){

				uint64_t next_items = levels[level-1][offset/32].unset_both_atomic(offset%32);

				uint64_t comp = ~READ_BOTH(offset%32);

				items = next_items & comp;

			} else {
				return;
			}



		} while(ascend(level, offset));

	}


	__device__ void correct_up_v3(int level, uint64_t offset, uint64_t items, int index){



		//uint64_t items = levels[level][offset].global_load_this();

		//if all bits unset
		if ((!(items & ~READ_BOTH(index))) && (level > 0)){

			uint64_t next_items = levels[level-1][offset/32].unset_both_atomic(offset%32);

			//uint64_t comp = ~READ_BOTH(offset%32);

			correct_up_v3(level-1, offset/32, next_items, offset % 32);

		}

	}

	//on free, set bits up, unless all the bits are set!
	__device__ void float_up_v2(int level, uint64_t offset, uint64_t items){



		//uint64_t items = levels[level][offset].global_load_this();

		//can't float up the top level
		if (level == 0) return;

		//clip by 32 to start
		offset = offset/32;

		//all items 1
		if ((!(~items))){

			uint64_t next_items = levels[level-1][offset/32].unset_both_atomic(offset%32);


			//need to unset? force full unset and then compare
			//need bit unset before
			if (levels[level][offset].try_swap_empty()){



					uint64_t next_items = levels[level-1][offset/32].set_both_atomic(offset%32) | READ_BOTH(offset%32);
					//set the level above
					float_up_v2(level-1,offset, next_items);

					//and reset
					levels[level][offset].swap_full();
					return;
	

			}

		} else if (__popcll(items) == 2){
			//set the bit above and return

			uint64_t next_items = levels[level-1][offset/32].set_control_bit_atomic(offset % 32) | SET_SECOND_BIT(offset%32);
			return float_up_v2(level-1, offset, next_items);

		}

		//if neither case, drop it.
		return;

	}



	__device__ void correct_empty(int level, uint64_t offset){


		if (level == 0) return;

		offset = offset/32;


		uint64_t items;



		do {


			items = levels[level-1][offset/32].set_control_bit_atomic(offset%32);

			ascend(level, offset);


		} while ((items == 0ULL) && (level != 0));

		return;



	}


	//needs to first unset bit, then swap out level below
	//on free, set bits up, unless all the bits are set!
	__device__ int float_up_v3(int level, uint64_t offset, uint64_t items, int index){


		//need to first clear upper bit, then unset all


		int next_index = offset % 32;

		offset = offset/32;

		if (level == 0) return 0;


		//first unset
		if (__popcll(~(items | READ_BOTH(index))) == 0){

			//unset next_items - next items is up to date.
			uint64_t next_items = levels[level-1][offset].unset_both_atomic(next_index) & (~READ_BOTH(next_index));


			if (levels[level][offset].unset_bits(items)){


				return 1 + float_up_v3(level-1, offset, next_items, next_index);



			} else {

				//on failure reset bit.
				levels[level-1][offset].set_control_bit_atomic(next_index);

			}



		}


		return 0;


	}


	__device__ void correct_float_up(int corrections, int level, uint64_t offset){

		while (corrections > 1){

			if (!levels[level][offset].set_bits(~0ULL)){

				printf("Fuckup in correction\n");

			}


			ascend(level, offset);


		}


		//undo last oneeeee

		if (!levels[level][offset].set_bits(~0ULL)){

			printf("Fuckup in correction\n");

		}


		levels[level][offset/32].set_both_atomic(offset%32);

	}


	//Needs to account for both 00 and 01 cases
	//float up, adjusting the overarching metadata to be accurate.
	//whenever we observe a layer that is full, we unset the previous layer.
	__device__ void correct_up(int level, uint64_t & offset, uint64_t & current_uint){



		//couple of cases.
		//if current_uint is full, we set to 11
		//else, we set to 01.
		//always floating the first bit though
		//if we set to 01 and it's already set, we are done!
		while (ascend(level, offset)){


			int index = offset % 32;

			if (~current_uint){

				//some bits set, set to 01
				current_uint = levels[level][offset/32].set_control_bit_atomic(index);

				//was already set
				if ((current_uint & READ_BOTH(index)) == SET_SECOND_BIT(index)){

					return;

				}


				current_uint |= SET_SECOND_BIT(index);

			} else {

				//some bits set and ascend by default
				current_uint = levels[level][offset/32].set_both_atomic(index);

				current_uint |= READ_BOTH(index);



			}


		}


	}


	__device__ void free(void * alloc){

		int level = num_levels-1;

		uint64_t offset = cast_to_offset(alloc);


		int index = offset%32;

		uint64_t items = levels[level][offset/32].set_both_atomic(offset%32);


		if (items == 0ULL) {

			correct_empty(level, offset);

		} else if (__popcll(~items) == 2){
			int count = float_up_v3(level, offset/32, items, index);


			if (count > 0) correct_float_up(count, level, offset);

			return;


		}


	}

	__device__ void free_old(void * alloc){


		int level = num_levels-1;

		uint64_t offset = cast_to_offset(alloc);


		//assert(offset < 32);

		cast_to_offset(alloc);

		//printf("Thread %llu dealing with offset %llu\n", threadIdx.x, offset);

		//return;

		while(true){


			uint64_t uint64_t_offset = offset/32;

			int index = offset % 32;

			//will be 11 at all points up till we hit 00
			//unless another thread is fighting us - which would be 01
			//so set second, if we succeed set first


			uint64_t old_val = levels[level][uint64_t_offset].set_both_atomic(index);

			//uint64_t old_val = levels[level][uint64_t_offset];


			if (!(old_val & READ_BOTH(index))){

				//successful update to 00! read up.

				old_val |= READ_BOTH(index);

				correct_up(level, offset, old_val);


				return;


			} else {

				if (!ascend(level, offset)){

					printf("Bitbuddy thread %llu ran into a double free at level %d offset %llu\n", threadIdx.x+blockIdx.x*blockDim.x, level, offset);
					asm("trap;");

				}
			}

		}

	}


	//instead of multiplying by 32 each time, we know that it is 2^5 so a large left shift is equivalent.
	__device__ uint64_t move_size_down(uint64_t size_in_bytes, int levels_down){
		return size_in_bytes << (5*levels_down);
	}

	//convert to the correct offset at the lowest level
	__device__ uint64_t cast_to_offset(void * alloc){



		uint64_t offset_at_lowest = move_size_down(bytes_at_top, num_levels-1);

		//check how many bytes difference, and correct to be the offset.
		return (((uint64_t) alloc) - ((uint64_t) memory))/offset_at_lowest;


	}





	//Precondition: Can only be called on valid pointers.
	//if we maintain this, we don't need to check anything
	__device__ uint64_t select_down(uint64_t current_level, int next_id){

		return (current_level*32+next_id);

	}

	__device__ uint64_t select_up(uint64_t current_level){
		//may
		return current_level/32;
	}

	//In malloc, we are looking for a selection of our size that is 11, and we want to convert it to 00
	//do this by setting it to 01 first 




	};





}

}


#endif //GPU_BLOCK_