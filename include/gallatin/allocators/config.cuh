#ifndef GALLATIN_CONFIG
#define GALLATIN_CONFIG


//shared includes for all files
// this allows us to use CUDA and cooperative groups API
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>



#define DO_PRAGMA(x) _Pragma (#x)

#define GALLATIN_TODO(x) DO_PRAGMA(message ("TODO - " #x))
//#define GALLATIN_TODO(x) _Pragma("message TODO - " #x)


namespace gallatin {

  //enable/disable debug operations
  // 0 is no_debug, 1 is debug.

  #ifndef GALLATIN_DEBUG
  #define GALLATIN_DEBUG 0
  #endif



  //block config parameters
  #define GALLATIN_ALLOCATIONS_PER_BLOCK 4096
  #define GALLATIN_BLOCK_TREE_OFFSET 26


  //error log config parameters



  #ifndef GALLATIN_ERROR_LOG_LENGTH
  #define GALLATIN_ERROR_LOG_LENGTH 100
  #endif


  //veb config

  //7 levels allows for 32*1024^3 objects even for the smallest case.
  //I'm going to say that this is a reasonable restriction for this use case
  //and that avoiding 1 indirection per lookup is more ideal.
  #define VEB_MAX_LEVELS 7


  //segment config
  //up to a this many threads can safely attempt a block request
  //packing in this way allows for one atomic to capture info
  // on both size and availability.
  // theoretically this saves an atomic over the previous implementation.
  #define SEGMENT_SIZE_OFFSET 2000000ULL
  #define SEGMENT_PACKED_COUNTER_OFFSET 48


  //block storage config
  //segment_id + block_id uniquely identify a block
  //given the limits on thiese, we can pack them into a uint32_t together

  #define BS_DEAD_BIT 0
  #define BS_SET_BIT 1
  #define BS_BLOCK_ID_CUTOFF 12


  //Full allocator parameters

  //define how many times a thread should attempt to fetch a new block before abandoning
  //lower numbers increase the chance a block will "fail" at high loads
  //but increase the time to mark failed at high load.

  #define GALLATIN_MAX_NEW_SLICE_ATTEMPTS 1000
  #define GALLATIN_MAX_NEW_BLOCK_ATTEMPTS 200
  #define GALLATIN_MAX_NEW_SEGMENT_ATTEMPTS 200

  #define GALLATIN_SEGMENT_SLEEP_DURATION 64


} //namespace gallatin



//weird placement but we want alloc utils to load after config
#include <gallatin/allocators/alloc_utils.cuh>

//and include error log for reporting.
#include <gallatin/allocators/internal_error_log.cuh>


#endif







