#ifndef GALLATIN_DS_UTILS
#define GALLATIN_DS_UTILS

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

//Data struct utils to perform atomics on templated types.
namespace gallatin {

namespace utils {


//These, of course, are atomics
//don't call these on stack variables

template<typename T>
__device__ __inline__ bool typed_atomic_write(T * backing, T item, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  static_assert(sizeof(T) > 8);

  uint64_t uint_item = ((uint64_t *) &item)[0];

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  if (typed_atomic_write<uint64_t>((uint64_t *) backing, uint_item, uint_replace)){

    //succesful? - flush write
    backing[0] = replace;
    return true;

  }

  return false;
}


template<>
__device__ __inline__ bool typed_atomic_write<uint16_t>(uint16_t * backing, uint16_t item, uint16_t replace){


  return (atomicCAS((unsigned short int *) backing, (unsigned short int) item, (unsigned short int) replace) == item);

}

// __device__ __inline__ bool typed_atomic_write<unsigned short>(unsigned short * backing, unsigned short item, unsigned short replace){


//  return (atomicCAS((unsigned short int *) backing, (unsigned short int) item, (unsigned short int) replace) == item);

// }



template<>
__device__ __inline__ bool typed_atomic_write<uint32_t>(uint32_t * backing, uint32_t item, uint32_t replace){


  return (atomicCAS((unsigned int *) backing, (unsigned int) item, (unsigned int) replace) == item);

}

template<>
__device__ __inline__ bool typed_atomic_write<uint64_t>(uint64_t * backing, uint64_t item, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return (atomicCAS((unsigned long long int *) backing, (unsigned long long int) item, (unsigned long long int) replace) == item);

}



template<typename T>
__device__ __inline__ T typed_atomic_CAS(T * backing, T item, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  //abort();

  static_assert(sizeof(T) > 8);

  uint64_t uint_item = ((uint64_t *) &item)[0];

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  uint64_t first_write = typed_atomic_CAS<uint64_t>((uint64_t *) backing, uint_item, uint_replace);

  if (first_write == uint_item){

    //succesful? - flush write
    backing[0] = replace;
    return first_write;

  }

  return first_write;
}


template<>
__device__ __inline__ uint16_t typed_atomic_CAS<uint16_t>(uint16_t * backing, uint16_t item, uint16_t replace){

  uint16_t result = atomicCAS((unsigned short int *) backing, (unsigned short int) item, (unsigned short int) replace);

  return result;

}


template<>
__device__ __inline__ uint32_t typed_atomic_CAS<uint32_t>(uint32_t * backing, uint32_t item, uint32_t replace){


  return atomicCAS((unsigned int *) backing, (unsigned int) item, (unsigned int) replace);

}

template<>
__device__ __inline__ uint64_t typed_atomic_CAS<uint64_t>(uint64_t * backing, uint64_t item, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return atomicCAS((unsigned long long int *) backing, (unsigned long long int) item, (unsigned long long int) replace);

}


template<typename T>
__device__ __inline__ T typed_atomic_exchange(T * backing, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  abort();

  static_assert(sizeof(T) > 8);

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  uint64_t first_write = typed_atomic_exchange<uint64_t>((uint64_t *) backing, uint_replace);

  return first_write;
}


// template<>
// __device__ __inline__ uint16_t typed_atomic_exchange<uint16_t>(uint16_t * backing, uint16_t replace){

//   uint16_t result = atomicExch((unsigned short int *) backing, (unsigned short int) replace);

//   return result;

// }


template<>
__device__ __inline__ uint32_t typed_atomic_exchange<uint32_t>(uint32_t * backing, uint32_t replace){


  return atomicExch((unsigned int *) backing, (unsigned int) replace);

}

template<>
__device__ __inline__ uint64_t typed_atomic_exchange<uint64_t>(uint64_t * backing, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return atomicExch((unsigned long long int *) backing, (unsigned long long int) replace);

}

template<>
__device__ __inline__ float typed_atomic_exchange<float>(float * backing, float replace){

  //printf("Uint64_t call being accessed\n");

  return atomicExch(backing, replace);

}

template<>
__device__ __inline__ int typed_atomic_exchange<int>(int * backing, int replace){

  //printf("Uint64_t call being accessed\n");

  return atomicExch(backing, replace);

}


//start of atomicAnd

template<typename T>
__device__ __inline__ T typed_atomic_and(T * backing, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  abort();

  static_assert(sizeof(T) > 8);

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  uint64_t first_write = typed_atomic_and<uint64_t>((uint64_t *) backing, uint_replace);

  return first_write;
}


// template<>
// __device__ __inline__ uint16_t typed_atomic_exchange<uint16_t>(uint16_t * backing, uint16_t replace){

//   uint16_t result = atomicExch((unsigned short int *) backing, (unsigned short int) replace);

//   return result;

// }


template<>
__device__ __inline__ uint32_t typed_atomic_and<uint32_t>(uint32_t * backing, uint32_t replace){


  return atomicAnd((unsigned int *) backing, (unsigned int) replace);

}

template<>
__device__ __inline__ uint64_t typed_atomic_and<uint64_t>(uint64_t * backing, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return atomicAnd((unsigned long long int *) backing, (unsigned long long int) replace);

}

//end of atomicAnd


//start of atomicOr

template<typename T>
__device__ __inline__ T typed_atomic_or(T * backing, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  abort();

  static_assert(sizeof(T) > 8);

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  uint64_t first_write = typed_atomic_or<uint64_t>((uint64_t *) backing, uint_replace);

  return first_write;
}


// template<>
// __device__ __inline__ uint16_t typed_atomic_exchange<uint16_t>(uint16_t * backing, uint16_t replace){

//   uint16_t result = atomicExch((unsigned short int *) backing, (unsigned short int) replace);

//   return result;

// }


template<>
__device__ __inline__ uint32_t typed_atomic_or<uint32_t>(uint32_t * backing, uint32_t replace){


  return atomicOr((unsigned int *) backing, (unsigned int) replace);

}

template<>
__device__ __inline__ uint64_t typed_atomic_or<uint64_t>(uint64_t * backing, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return atomicOr((unsigned long long int *) backing, (unsigned long long int) replace);

}

//end of atomicOr

//start of atomicAdd

template<typename T>
__device__ __inline__ T typed_atomic_add(T * backing, T replace){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  abort();

  static_assert(sizeof(T) > 8);

  uint64_t uint_replace = ((uint64_t *) &replace)[0];

  uint64_t first_write = typed_atomic_add<uint64_t>((uint64_t *) backing, uint_replace);

  return first_write;
}


// template<>
// __device__ __inline__ uint16_t typed_atomic_exchange<uint16_t>(uint16_t * backing, uint16_t replace){

//   uint16_t result = atomicExch((unsigned short int *) backing, (unsigned short int) replace);

//   return result;

// }


template<>
__device__ __inline__ uint32_t typed_atomic_add<uint32_t>(uint32_t * backing, uint32_t replace){


  return atomicAdd((unsigned int *) backing, (unsigned int) replace);

}

template<>
__device__ __inline__ uint64_t typed_atomic_add<uint64_t>(uint64_t * backing, uint64_t replace){

  //printf("Uint64_t call being accessed\n");

  return atomicAdd((unsigned long long int *) backing, (unsigned long long int) replace);

}

template<>
__device__ __inline__ int typed_atomic_add<int>(int * backing, int replace){


  return atomicAdd((int *) backing, (int) replace);

}

//end of atomicAdd


template<typename T>
__device__ __inline__ T typed_global_read(T * backing){


  //atomic CAS first bit

  //this should break, like you'd expect it to
  //TODO come back and make this convert to uint64_t for CAS
  //you can't CAS anything smaller than 16 bits so I'm not going to attempt that

  //printf("I am being accessed\n");

  abort();

  static_assert(sizeof(T) > 8);

  uint64_t first_write = typed_global_read<uint64_t>((uint64_t *) backing);

  return first_write;
}


template<>
__device__ __inline__ uint16_t typed_global_read<uint16_t>(uint16_t * backing){

  uint16_t result = gallatin::utils::global_read_uint16_t(backing);

  return result;

}


template<>
__device__ __inline__ uint32_t typed_global_read<uint32_t>(uint32_t * backing){

  uint32_t result = gallatin::utils::ldca((uint *) backing);

  return result;
}

template<>
__device__ __inline__ int typed_global_read<int>(int * backing){

  int result = (int) gallatin::utils::ldca((uint *) backing);

  return result;
}

template<>
__device__ __inline__ uint64_t typed_global_read<uint64_t>(uint64_t * backing){

  //printf("Uint64_t call being accessed\n");

  uint64_t result = gallatin::utils::ldca(backing);

  return result;

}



}  // namespace utils

}  // namespace gallatin

#endif  // GPU_BLOCK_