/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef _HASHUTIL_CUH_
#define _HASHUTIL_CUH_

#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

__host__ __device__ uint64_t MurmurHash64B ( const void * key, int len, unsigned int seed );
__host__ __device__ uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed );

__host__ __device__ uint64_t hash_64(uint64_t key, uint64_t mask);
__host__ __device__ uint64_t hash_64i(uint64_t key, uint64_t mask);

#endif  // #ifndef _HASHUTIL_H_


