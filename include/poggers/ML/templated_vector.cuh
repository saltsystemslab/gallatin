#ifndef POGGERS_VECTOR
#define POGGERS_VECTOR


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

//#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace ml {






//Sparse neuron
//contains weights, activation, bias, and an activation function

//templateable type gives support for bizarre scalars and reduced float16 if we want it.
//does cuda natively support?
template <uint64_t size, typename scalar_type>
struct vector {


	scalar_type weights[size];



};

}

}


#endif //Layer