#ifndef REP_HELPERS 
#define REP_HELPERS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace helpers {


template<typename T>
struct return_
{
    typedef T type;
};

//given bytes used generate the main
template<uint64_t N>
struct bytetype : return_<uint64_t> {};

template<>
struct bytetype<4> : return_<uint32_t> {};

template<>
struct bytetype<3> : return_<uint32_t> {};

template<>
struct bytetype<2> : return_<uint16_t> {};

template<>
struct bytetype<1> : return_<uint8_t> {};


template <typename T>
__device__ __inline__ bool get_evenness_fast(T item){
	return item & 1;
}



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


// 	return (atomicCAS((unsigned short int *) backing, (unsigned short int) item, (unsigned short int) replace) == item);

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

	abort();

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


//pushed lower since it utilizes typed atomic write

//if we can get away with this templated do so
//we know keys only separate on well defined addresses so prebaking would be better
//returns true if partial insert succeeds, false else
//can be templated with a storage type that satisfies the requirements of typed_atomic_write.
template <typename storage_type, typename key_type>
__device__ inline bool typed_atomic_sub_write(storage_type * address_to_swap, key_type expected, key_type replacement, int offset, int num_bits){


	//if this isn't true, write must be split
	//static_assert(sizeof(storage_type) < offset+num_bits);

	//test that key is only num_bits
	//potential problem here when num bits == sizeof(key_type)*4


	//if you do this you're a dumbass, just use a lossless SOA type
	//assert(num_bits != sizeof(key_type*4));

	//mask that defines only the lower num_bits bits
	key_type KEY_MASK = (1ULL << num_bits) -1;

	//assert that no extra bits are present
	//it's a lot easier in higher order functions if this is not taken as a precondition
	//instead we can use the KEY_MASK to guarantee that no more than num_bits bits are written
	//so the replacement_key can be passed to this function untouched and then degraded via a left shift for any remainder.
	//assert(replacement & KEY_MASK == replacement);


	//at this point the key_type has only the lower num_bits set
	//prep key of expected

	//printf("Address to swap: %p\n", address_to_swap);


	while (true){


		storage_type read = address_to_swap[0];







		//if the bits stored are already set

		storage_type replaced_bits ((read >> offset) & KEY_MASK);
		if (replaced_bits != expected){

			//return those bits

			//cast up to key_type for consistency
			//all unsigned types, should be fine.
			return false;

		}


		//othewise we're clear to attempt


		//need to combine two types

		//00000____00000;
		//and ____XXXX____;

		//first mask read
		//then & with (replacement << offset)?
		storage_type new_read = read & (~(KEY_MASK << offset));

		//at this point new read is everything but the empty section


		storage_type expected_read = new_read | (expected << offset);

		new_read = new_read | ((replacement & KEY_MASK) << offset);



		if (typed_atomic_CAS<storage_type>(address_to_swap, expected_read, new_read) == expected_read) return true;


	}

	





}


//given an array of length Size_of_array that contains tightly packed bits, check them against a queried key.
template <typename storage_type, typename key_type, std::size_t Size_of_array>
__device__ storage_type sub_byte_contains(storage_type * address_to_swap, int index_into_list, int num_bits){

	assert(index_into_list < Size_of_array);

	//constant size of the storage type, expressed in bits
	const int size_of_storage_bits = sizeof(storage_type)*8;

	//what git does the item we want to read start at
	const int start_bit_write = index_into_list*num_bits;

	//the last bit needed for this item
	const int end_bit_write = (index_into_list+1)*num_bits-1;

	//modulus of the size of the storage bits gives us the offset from a storage object.
	const int offset = start_bit_write % size_of_storage_bits;

	//these two calculations store the addresses into the list of the start and end 
	// if they are the same we can perform a static read.

	const int start_index = start_bit_write/size_of_storage_bits;

	const int end_index = end_bit_write/size_of_storage_bits;

	if (start_index == end_index){


		//define a mask to extract the data
		//we are starting at start_index + offset_bits, running num_bits extra
		//TODO: double check endian correctness - I'm afraid im backwards in cuda.
		//this *shouldn't* matter as we at no point read a contiguous section
		//but better safe than sorry.


		const storage_type KEY_MASK_SET_BITS = (1ULL << num_bits)-1;
		const storage_type KEY_MASK = (KEY_MASK_SET_BITS << offset);

		key_type return_key = (key_type) (address_to_swap[start_index] & KEY_MASK) >> offset;

		return return_key;

	} else {


		//prep two masks

		//first is the KEY_MASK, which only covers upper bits
		//double check if there is a more efficient way to set this
		//1ULL << bits_to_set - 1 << 0s after is three operations, you've gotta be able to beat that
		//hopefully done at compile time? I need to get better at godbolting these things 

		//to prep mask we need to know how many bits run over
		const int leftover_bits = (offset+num_bits) % size_of_storage_bits;

		//if these two don't match our assumptions are super weird
		assert(num_bits + offset - leftover_bits == size_of_storage_bits);

		const storage_type KEY_MASK_SET_BITS = (1ULL << num_bits-leftover_bits)-1;
		const storage_type KEY_MASK = (KEY_MASK_SET_BITS << offset);

		//front section is the lower order bits - this all seems like good business to me
		storage_type front_section = (address_to_swap[start_index] & KEY_MASK) >> offset;



		//key_mask_should_be

		//bitshift and not?

		//set the uppermost bit, then -1 and ~
		//that does the trick IFF ~ is cheaper than <<

		const storage_type LOWER_MASK = (1ULL << leftover_bits)-1;

		const storage_type read_value = address_to_swap[start_index + 1] & LOWER_MASK;

		//this isn't offset


		storage_type lower_section = read_value << (num_bits - leftover_bits);

		//return the union of the sections
		return front_section | lower_section;



	}



}


	
//given an array whose head is properly aligned,
//perform a fucky misalgined write into the array
template <typename storage_type, typename key_type, std::size_t Size_of_array>
__device__ inline bool sub_byte_atomic_write(storage_type * address_to_swap, key_type replacement_key, int index_into_list, int num_bits){

	assert(index_into_list < Size_of_array);


	const int size_of_storage_bits = sizeof(storage_type)*8;

	const int start_bit_write = index_into_list*num_bits;

	const int end_bit_write = (index_into_list+1)*num_bits-1;

	//this is always needed, start of first write
	const int offset = start_bit_write % size_of_storage_bits;

	//need # bits in each of the sub partitions

	const int storage_address_offset = start_bit_write/size_of_storage_bits;

	//for the last write - 
	//end bit_write = 

	//printf("Storage address offset %d\n", storage_address_offset);

	//printf("Start address, back address %d, %d", start_bit_write/size_of_storage_bits, end_bit_write/size_of_storage_bits);

	//TODO: constexpr if this
	if (start_bit_write/size_of_storage_bits != end_bit_write/size_of_storage_bits){

		//printf("Double Atomic setup\n");

		//printf("Start bit: %d, end bit: %d\n", start_bit_write, end_bit_write);

		const int leftover_bits = (offset+num_bits) % size_of_storage_bits;

		const int internal_num_bits = num_bits - leftover_bits;


		//offset is the expected offset into the given address
		//for secondary writes this must be 0 as the write extends from a previous boundary to the edge of the current container,
		if (typed_atomic_sub_write<storage_type, key_type>(address_to_swap + storage_address_offset, 0, replacement_key, offset, internal_num_bits)){

			bool result = typed_atomic_sub_write<storage_type, key_type>(address_to_swap+storage_address_offset+1, 0, replacement_key >> internal_num_bits, 0, leftover_bits);

			assert(result == true);

			//if this write succeeded the secondary *must* go through.
			return true;

		}

		return false;



	} else {

		//printf("Single Atomic Setup\n");

		//aligned one pass write
		return typed_atomic_sub_write<storage_type, key_type>(address_to_swap + storage_address_offset, 0, replacement_key, offset, num_bits);


	}



}



//TODO:: atomic read
//attempts to replace a uint16_t based on a mask
//returns success if the replacement of a subsection succeeds
//false otherwise
__device__ inline bool replace_uint16(uint16_t * address, uint16_t expected, uint16_t desired, uint16_t mask){

	//expected/desired should be configured to work with mask

	while (true){

		//atomicAdd((unsigned short int * ) 0)

		//set bits that should be masked to 0

		uint16_t read_from_address = address[0] & ~mask;
		// atomicCAS((unsigned short * ) address, (unsigned short) 0, (unsigned short) 0) & ~mask;
		//uint16_t read_from_address = address[0] & ~mask;

		uint16_t expected_and_mask = expected & mask;

		uint16_t expected_read = expected_and_mask | read_from_address;

		//get mask for desired

		uint16_t desired_mask = desired & mask; 

		uint16_t desired_read = desired_mask | read_from_address;



		uint16_t result = typed_atomic_CAS<uint16_t>(address, expected_read, desired_read);

		__threadfence();

		if (result == expected_read) return true;

		//this is only true if our expectation was wrong
		if ((result & (~mask)) == read_from_address) return false;

	}



}

	
//for the second half of any two part insert, an atomicOR operation will suffice
//this should hopefully up the throughput of these sections over performing two atomicCAS operations
// __device__ inline bool atomicOr_uint16(uint16_t * address, uint16_t expected, uint16_t desired, uint16_t mask){

// 	//expected/desired should be configured to work with mask


// 		//atomicAdd((unsigned short int * ) 0)

// 		//set bits that should be masked to 0

// 		//uint16_t read_from_address = address[0] & ~mask;
// 		// atomicCAS((unsigned short * ) address, (unsigned short) 0, (unsigned short) 0) & ~mask;
// 		//uint16_t read_from_address = address[0] & ~mask;

// 		//uint16_t expected_and_mask = expected & mask;

// 		//uint16_t expected_read = expected_and_mask | read_from_address;

// 		//get mask for desired

// 		uint16_t desired_mask = desired & mask; 

// 		//uint16_t desired_read = desired_mask | read_from_address;



// 		uint16_t result = typed_atomic_CAS<uint16_t>(address, expected_read, desired_read);

// 		__threadfence();

// 		if (result == expected_read) return true;

// 		//this is only true if our expectation was wrong
// 		if ((result & (~mask)) == read_from_address) return false;




// }


template<>
__device__ __inline__ bool typed_atomic_write<uint8_t>(uint8_t * backing, uint8_t item, uint8_t replace){


	uint64_t offset = (uint64_t) backing;

	uint16_t mask;

	uint16_t * address_to_write;

	uint16_t expected;

	uint16_t desired;


	if ((offset % 2) == 0){

		mask = (1ULL << 8)-1;

		address_to_write = (uint16_t * ) backing;

		expected = item;

		desired = replace;

		assert((expected & mask ) == item);
		assert((desired & mask) == replace);

	} else {

		mask = (1ULL << 8)-1;

		mask = mask << 8;


		expected = item;

		expected = expected << 8;

		desired = replace;

		desired = replace << 8;

		assert ((expected & mask) == expected);
		assert ((desired & mask) == desired);

		assert (((desired & mask) >> 8) == replace);
		assert (((expected & mask) >> 8) == item);


		address_to_write = (uint16_t * ) (backing-1);




	}




	return replace_uint16(address_to_write, expected, desired, mask);

}


__device__ inline uint16_t read_uint16(uint16_t * address, uint16_t mask){

	return address[0] & mask;

}

__device__ inline uint16_t read_first(uint16_t * address){

	uint16_t mask = (1ULL << 12)-1;

	return read_uint16(address, mask);

}

__device__ inline uint16_t read_second(uint16_t * address){

	uint16_t first_mask = (1ULL << 4)-1;

	first_mask = first_mask << 12;

	uint16_t second_mask = (1ULL << 8) -1;

	uint16_t ret_val = 0;
	ret_val |= read_uint16(address, first_mask);
	ret_val |= read_uint16(address+1, second_mask);

	return ret_val;


}

__device__ inline uint16_t read_third(uint16_t * address){

	uint16_t first_mask = (1ULL << 8)-1;

	first_mask = first_mask << 8;

	uint16_t second_mask = (1ULL << 4) -1;

	uint16_t ret_val = 0;
	ret_val |= read_uint16(address+1, first_mask);
	ret_val |= read_uint16(address+2, second_mask);

	return ret_val;


}

__device__ inline uint16_t read_fourth(uint16_t * address){

	uint16_t mask = (1ULL << 12)-1;

	mask = mask << 4;

	return read_uint16(address+2, mask);

}


__device__ inline bool empty_first(uint16_t * address){

	uint16_t mask = (1ULL << 12)-1;

	//uint16_t masked_item = item & mask;

	

	return (0 == read_uint16(address, mask));

}

__device__ inline bool empty_second(uint16_t * address){

	uint16_t first_mask = (1U << 4)-1;

	first_mask = first_mask << 12;

	uint16_t second_mask = (1U << 8) -1;

	uint16_t ret_val = 0;

	//address[0] & mask;
	ret_val |= read_uint16(address, first_mask);
	ret_val |= read_uint16(address+1, second_mask);

	return (ret_val == 0);


}

__device__ inline bool empty_third(uint16_t * address){

	uint16_t first_mask = (1ULL << 8)-1;

	first_mask = first_mask << 8;

	uint16_t second_mask = (1ULL << 4) -1;

	uint16_t ret_val = 0;
	ret_val |= read_uint16(address+1, first_mask);
	ret_val |= read_uint16(address+2, second_mask);

	

	return (ret_val == 0);


}

__device__ inline bool empty_fourth(uint16_t * address){

	uint16_t mask = (1ULL << 12)-1;

	mask = mask << 4;

	//uint16_t masked_item = item & mask;

	//if (masked_item == 0) masked_item++;

	return (0 == read_uint16(address+2, mask));

}


__device__ inline bool match_first(uint16_t * address, uint16_t item){

	uint16_t mask = (1ULL << 12)-1;

	uint16_t masked_item = item & mask;

	if (masked_item == 0) masked_item++;

	return masked_item == read_uint16(address, mask);

}

__device__ inline bool match_second(uint16_t * address, uint16_t item){

	uint16_t first_mask = (1ULL << 4)-1;

	first_mask = first_mask << 12;

	uint16_t second_mask = (1ULL << 8) -1;

	uint16_t ret_val = 0;

	//address[0] & mask;
	//00...0111
	ret_val |= read_uint16(address, first_mask);
	ret_val |= read_uint16(address+1, second_mask);


	uint16_t masked_item = (item & first_mask) | (item & second_mask);

	if ((item & first_mask) == 0){
		masked_item |= (1ULL << 12);
	}
	//uint16_t masked_item = item_mask & item;

	//if (masked_item == 0) m;

	return (ret_val == masked_item);


}

__device__ inline bool match_third(uint16_t * address, uint16_t item){

	uint16_t first_mask = (1ULL << 8)-1;

	first_mask = first_mask << 8;

	uint16_t second_mask = (1ULL << 4) -1;

	uint16_t ret_val = 0;
	ret_val |= read_uint16(address+1, first_mask);
	ret_val |= read_uint16(address+2, second_mask);

	uint16_t masked_item = (item & first_mask) | (item & second_mask);

	if ((item & first_mask) == 0){
		masked_item |= (1ULL << 8);
	}

	return (ret_val == masked_item);


}

__device__ inline bool match_fourth(uint16_t * address, uint16_t item){

	uint16_t mask = (1ULL << 12)-1;

	mask = mask << 4;

	uint16_t masked_item = item & mask;

	if (masked_item == 0) masked_item |= (1ULL << 4);

	return (masked_item == read_uint16(address+2, mask));

}



//These functions swap a uint16_t 12 bit pointer
//aligned to the next 3 uint16_t, containing 4 items each
//always called from the first of the 3
__device__ inline bool replace_first(uint16_t * address, uint16_t expected, uint16_t desired){

	
	uint16_t mask = (1ULL << 12)-1;

	if ((desired & mask) == 0){

		desired++;
		assert((desired & mask) == 1);



	} 


	// desired = desired | 1ULL;

	return replace_uint16(address, expected, desired, mask);


}

//replaces upper 4 of 0, lower 8 of 1
__device__ inline bool replace_second(uint16_t * address, uint16_t expected, uint16_t desired){


	uint16_t first_mask = (1ULL << 4)-1;

	//bitshift up 12  
	first_mask = first_mask << 12;

	uint16_t second_mask = (1ULL << 8)-1;


	if ((desired & first_mask) == 0){

		desired |= (1ULL << 12);

	}


	//printf("Desired %u, desired front masked: %u, desired second mask: %u, correction %u, desired corrected: %u\n", desired, desired & first_mask, desired & second_mask, (1ULL << 12), (desired | (1ULL << 12)) & first_mask);




	//0000 
	if (replace_uint16(address, expected, desired, first_mask)){


		bool replaced = replace_uint16(address+1, expected, desired, second_mask);

		
		assert(replaced);

		return true;

	}

	return false;


}

//upper 8 of 1, lower 4 of 2
__device__ inline bool replace_third(uint16_t * address, uint16_t expected, uint16_t desired){

	uint16_t first_mask = (1ULL << 8)-1;

	//bitshift up 8
	first_mask = first_mask << 8;

	uint16_t second_mask = (1ULL << 4)-1;



	if ((desired & first_mask) == 0){

		desired |= (1ULL << 8);

	}


	if (replace_uint16(address+1, expected, desired, first_mask)){

		bool replaced = replace_uint16(address+2, expected, desired, second_mask);


		assert(replaced);

		return true;

	}

	return false;


}

//upper 12 of 2
__device__ inline bool replace_fourth(uint16_t * address, uint16_t expected, uint16_t desired){

	uint16_t mask = (1ULL << 12)-1;

	mask = mask << 4;

	if ((desired & mask) == 0){

		desired |= (1ULL<<4);
		assert((desired & mask) == (1ULL<<4));

	} 


	return replace_uint16(address+2, expected, desired, mask);

}

template <typename T>
__device__ inline bool sub_byte_match(uint16_t * address, T item, int index_to_access){



	uint16_t * address_to_check = address + 3*(index_to_access/4);

	int offset = index_to_access % 4;

	uint16_t short_item = (uint16_t) item;

	if (offset == 0){

		return match_first(address_to_check, short_item);

	} else if (offset == 1){

		return match_second(address_to_check, short_item);

	} else if (offset == 2){

		return match_third(address_to_check, short_item);

	} else {

		return match_fourth(address_to_check, short_item);
	}


}

__device__ inline bool sub_byte_empty(uint16_t * address, int index_to_access){

	uint16_t * address_to_check = address + 3*(index_to_access/4);

	int offset = index_to_access % 4;

	if (offset == 0){
		return empty_first(address_to_check);
	} else if (offset == 1){
		return empty_second(address_to_check);
	} else if (offset == 2){
		return empty_third(address_to_check);
 	} else {
 		return empty_fourth(address_to_check);
 	}

}

template <typename T> 
__device__ inline bool sub_byte_replace(uint16_t * address, T expected, T item, int index_to_access){

	uint16_t * address_to_check = address + 3*(index_to_access/4);

	int offset = index_to_access % 4;

	uint16_t short_expected = (uint16_t) expected;

	uint16_t short_item = (uint16_t) item;


	if (offset == 0){

		return replace_first(address_to_check, short_expected, short_item);

	} else if (offset == 1){

		return replace_second(address_to_check, short_expected, short_item);

	} else if (offset == 2) {

		return replace_third(address_to_check, short_expected, short_item);

	} else {

		return replace_fourth(address_to_check, short_expected, short_item);

	}

}



}

}


#endif //GPU_BLOCK_