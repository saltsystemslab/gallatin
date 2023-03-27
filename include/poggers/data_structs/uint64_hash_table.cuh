#ifndef POGGERS_UINT64_HT
#define POGGERS_UINT64_HT


#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>


#include <poggers/representations/dynamic_container.cuh>
#include <poggers/representations/key_only.cuh>

#include <poggers/sizing/default_sizing.cuh>
#include <poggers/sizing/variadic_sizing.cuh>
#include <poggers/tables/base_table.cuh>


//64 Bit Key-Val HT
//This is useful for a lot of projects, particularly those that map to or from pointers, as you can typecast in and out to their type.
//This table is intended for use in the allocators/swap_space, where it provides symbolic links from uint64_t -> pointer
// That being said, it is an SOL table (~1.6 Giga inserts on a V100), so it should be suitable for use in other applications.
//If you need a different size, you can build your own table from poggers::tables::static_table and set the template args yourself.

namespace poggers {

namespace data_structs {

	using hash_table_64 = poggers::tables::static_table<uint64_t,uint64_t, poggers::representations::key_val_pair, 4, 16, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

}


}


#endif //64 bit HT