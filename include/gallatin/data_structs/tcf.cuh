#ifndef POGGERS_TCF
#define POGGERS_TCF


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


namespace poggers {

namespace data_structs {

	using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
	using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;



}


}


#endif //Poggers TCF guard