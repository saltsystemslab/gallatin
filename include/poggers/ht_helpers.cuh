#ifndef HELPERS_H 
#define HELPERS_H


#include <cuda.h>
#include <cuda_runtime_api.h>


#include <assert.h>

//thrust stuff
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs

// template <typename Tag_type>
// __device__ bool assert_sorted(Tag_type * tags, int nitems){


// 	if (nitems < 1) return true;

// 	Tag_type smallest = tags[0];

// 	for (int i=1; i< nitems; i++){

// 		if (tags[i] < smallest) return false;

// 		smallest = tags[i];
// 	}

// 	return true;

// }

//specialized atomic_CAS


namespace poggers {

	class PoggersException : public std::exception {
    private:
    char * message;

    public:
    PoggersException(char * msg) : message(msg) {}
    char * what () const throw() d {
        return message;
    }
};

}

#endif //GPU_BLOCK_