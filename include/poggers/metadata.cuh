#ifndef METADATA
#define METADATA

#define DEBUG_ASSERTS 0
//#define MAX_FILL 28
#define SINGLE_REGION 0

//do blocks assume exclusive access? if yes, no need to lock
//this is useful for batched scenarios.
#define EXCLUSIVE_ACCESS 1


//number of warps launched per grid block
#define WARPS_PER_BLOCK 1
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)

#define BLOCKS_PER_THREAD_BLOCK 128

//# of blocks to be inserted per warp in the bulked insert phase
//#define REGIONS_PER_WARP 8


//power of 2 metadata
//#define POWER_BLOCK_SIZE 1024
//#define TOMBSTONE 1000000000000ULL
#define TOMBSTONE_VAL 0

#define SLOTS_PER_CONST_BLOCK 32



//Atomic blocks stats

#define TAG_BITS 16
#define VAL_BITS 16 

#define BYTES_PER_CACHE_LINE 128
#define CACHE_LINES_PER_BLOCK 2


//what kind of hashing? 
// 0 linear
// 1 quadratic
// 2 double
#define HASH_TYPE 2

//#define PARTITION_SIZE 4
#define ELEMS_PER_PARTITION 32


//metadata for hash tables
#define MAX_SLOTS_TO_PROBE 20
#define HT_EMPTY 0


#endif