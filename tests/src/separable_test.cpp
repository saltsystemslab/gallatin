/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file contains speed tests for several Hash Table Types
 *          built using POGGERS. For more verbose testing please see the 
 *          benchmarks folder.
 *
 * ============================================================================
 */




//#include "include/templated_quad_table.cuh"
#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/sizing/variadic_sizing.cuh>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>


using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
using tcqf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;


int main(int argc, char** argv) {

   poggers::sizing:size_in_num_slots<2> test(1000,100);

   tcqf * my_tcqf = tcqf::generate_on_device(&test, 120);

   tcqf::free_on_device(my_tcqf);

	return 0;

}
