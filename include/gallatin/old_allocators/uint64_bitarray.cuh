#ifndef POGGERS_UINT64_BITARRAY
#define POGGERS_UINT64_BITARRAY

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/free_list.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// BITMASK Stuff
//  #define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
//  #d efine BITMASK(nbits)                                    \
//   ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))
//  #define SET_BIT_MASK(index) ((1ULL << (index)))
#define UNSET_BIT_MASK(index) (~SET_BIT_MASK(index))

#define SET_FIRST_BIT(index) ((SET_BIT_MASK(index * 2 + 1)))
#define SET_SECOND_BIT(index) ((SET_BIT_MASK(index * 2)))

#define UNSET_FIRST_BIT(index) (~SET_FIRST_BIT(index))
#define UNSET_SECOND_BIT(index) (~SET_SECOND_BIT(index))

#define READ_BOTH(index) ((3ULL << (2 * index)))

// selects 01 from every pair
#define FULL_BIT_MASK 0x5555555555555555ULL

// selects 01
#define HAS_CONTROL_BIT_MASK 0x5555555555555555ULL

// selects 10 from every pair
#define HAS_CHILD_BIT_MASK 0xAAAAAAAAAAAAAAAAULL

// a pointer list managing a set section of device memory
namespace poggers {

namespace allocators {

struct uint64_t_bitarr {
  uint64_t bits;

  __host__ __device__ uint64_t_bitarr() { bits = 0ULL; }

  __host__ __device__ uint64_t_bitarr(uint64_t ext_bits) { bits = ext_bits; }

  // This only matches operands
  __device__ int get_random_unset_bit_full() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    // these two do the same calculation, but the upper uses half the ops!
    uint64_t both = ((~bits) & (~bits >> 1)) & FULL_BIT_MASK;
    // uint64_t both = ((bits & FULL_BIT_MASK) >> 1) & (bits &
    // HAS_CHILD_BIT_MASK);

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((both) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(both) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit_full() {
    // uint64_t tid = threadIdx.x*threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((threadIdx.x * 1610612741ULL + threadIdx.x) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    // these two do the same calculation, but the upper uses half the ops!
    uint64_t both = (bits & (bits >> 1)) & FULL_BIT_MASK;
    // uint64_t both = ((bits & FULL_BIT_MASK) >> 1) & (bits &
    // HAS_CHILD_BIT_MASK);

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((both) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(both) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  // find first active bit that is 11. Returns -1 if the index is not found
  __device__ int get_first_active_bit_full() {
    // quick shortcut to find
    uint64_t both = (bits & (bits >> 1)) & FULL_BIT_MASK;

    return __ffsll(both) - 1;
  }

  __device__ int get_random_active_bit_child() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t both = (bits & HAS_CHILD_BIT_MASK);
    // uint64_t both = ((bits & FULL_BIT_MASK) >> 1) & (bits &
    // HAS_CHILD_BIT_MASK);

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((both) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(both) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit_control_only() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t both =
        ((~(bits & HAS_CHILD_BIT_MASK) >> 1)) & (bits & HAS_CONTROL_BIT_MASK);

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((both) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(both) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit_control() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t both = (bits & HAS_CONTROL_BIT_MASK);
    // uint64_t both = ((bits & FULL_BIT_MASK) >> 1) & (bits &
    // HAS_CHILD_BIT_MASK);

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((both) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(both) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  // find first active bit that is 10 | 11. Returns -1 if the index is not found
  __device__ int get_first_active_bit_child() {
    // quick shortcut to find
    // uint64_t both = (bits & (bits >>1)) & FULL_BIT_MASK;

    return __ffsll(bits & HAS_CHILD_BIT_MASK) - 1;
  }

  __device__ int get_first_active_bit_control() {
    // quick shortcut to find
    // uint64_t both = (bits & (bits >>1)) & FULL_BIT_MASK;

    return __ffsll(bits & HAS_CONTROL_BIT_MASK) - 1;
  }

  __device__ int get_random_active_bit_nonzero() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((bits & ~SET_BIT_MASK(0)) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(bits & ~SET_BIT_MASK(0)) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll(bits & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(bits) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit_warp() {
    uint64_t tid = threadIdx.x / 32;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll(bits & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(bits) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_active_bit_warp_nonzero() {
    uint64_t tid = threadIdx.x / 32;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((bits & ~SET_BIT_MASK(0)) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(bits & ~SET_BIT_MASK(0)) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_random_unset_bit() {
    uint64_t tid = threadIdx.x * threadIdx.x;

    // big ol prime *1610612741ULL
    int random_cutoff = ((tid * 1610612741ULL) % 64);

    // does this mask need to check on 64 bit case?
    // no actually cause there is no functional difference as its just the last
    // bit?

    uint64_t random_mask = ((1ULL << random_cutoff) - 1);

    int valid_upper = __ffsll((~bits) & (~random_mask)) - 1;

    if (valid_upper != -1) {
      return valid_upper;
    }

    // upper bits are not set (from above) so we can save an op for __ffsll and
    // find first set for whole thing.
    return __ffsll(~bits) - 1;

    // uint64_t random_mask = ((1ULL << random_cutoff) -1);

    // int valid_upper = __ffsll(bits & (~random_mask)) -1;

    // if (valid_upper != -1){
    // 	return valid_upper;
    // }

    // //upper bits are not set (from above) so we can save an op for __ffsll
    // and find first set for whole thing. return __ffsll(bits) -1;
  }

  __device__ int get_first_active_bit() { return __ffsll(bits) - 1; }

  __device__ void invert() { bits = ~bits; }

  __device__ inline uint64_t generate_set_mask(int index) {
    return (1ULL) << index;
  }

  __device__ inline uint64_t generate_unset_mask(int index) {
    return ~generate_set_mask(index);
  }

  __device__ bool read_first_bit(int index) {
    return bits & SET_FIRST_BIT(index);
  }

  __device__ bool read_second_bit(int index) {
    return bits & SET_SECOND_BIT(index);
  }

  __device__ inline bool set_read_first_bit_atomic(int index) {
    uint64_t old =
        atomicOr((unsigned long long int *)this, SET_FIRST_BIT(index));

    return old & SET_FIRST_BIT(index);
  }

  __device__ inline uint64_t set_first_bit_atomic(int index) {
    return atomicOr((unsigned long long int *)this, SET_FIRST_BIT(index));
  }

  __device__ inline uint64_t set_lock_bit_atomic(int index) {
    return atomicOr((unsigned long long int *)this, SET_FIRST_BIT(index));
  }

  __device__ inline uint64_t set_both_atomic(int index) {
    return atomicOr((unsigned long long int *)this, READ_BOTH(index));
  }

  __device__ inline uint64_t unset_lock_bit_atomic(int index) {
    return atomicAnd((unsigned long long int *)this, ~SET_FIRST_BIT(index));
  }

  __device__ inline uint64_t set_control_bit_atomic(int index) {
    return atomicOr((unsigned long long int *)this, SET_SECOND_BIT(index));
  }

  __device__ inline uint64_t unset_control_bit_atomic(int index) {
    return atomicAnd((unsigned long long int *)this, ~SET_SECOND_BIT(index));
  }

  __device__ inline uint64_t unset_both_atomic(int index) {
    return atomicAnd((unsigned long long int *)this, ~READ_BOTH(index));
  }

  __device__ inline uint64_t set_second_bit_atomic(int index) {
    return atomicOr((unsigned long long int *)this, SET_SECOND_BIT(index));
  }

  __device__ inline bool set_read_second_bit_atomic(int index) {
    uint64_t old =
        atomicOr((unsigned long long int *)this, SET_SECOND_BIT(index));

    return old & SET_FIRST_BIT(index);
  }

  __device__ inline bool unset_read_first_bit_atomic(int index) {
    uint64_t old =
        atomicAnd((unsigned long long int *)this, ~SET_FIRST_BIT(index));

    return old & SET_FIRST_BIT(index);
  }

  __device__ inline uint64_t unset_flip_mask(uint64_t mask) {
    return atomicAnd((unsigned long long int *)this, ~mask);
  }

  __device__ inline bool unset_first_bit_atomic(int index) {
    return atomicAnd((unsigned long long int *)this, ~SET_FIRST_BIT(index));
  }

  __device__ inline bool unset_read_second_bit_atomic(int index) {
    uint64_t old =
        atomicAnd((unsigned long long int *)this, ~SET_SECOND_BIT(index));

    return old & SET_SECOND_BIT(index);
  }

  __device__ inline uint64_t unset_second_bit_atomic(int index) {
    return atomicAnd((unsigned long long int *)this, ~SET_SECOND_BIT(index));
  }

  __device__ inline void reset_both_atomic(uint64_t old, int index) {
    atomicOr((unsigned long long int *)this, READ_BOTH(index) & old);
  }

  __device__ bool set_bit_atomic(int index) {
    uint64_t set_mask = generate_set_mask(index);

    uint64_t old = atomicOr((unsigned long long int *)this, set_mask);

    // old should be empty

    return (~old & set_mask);
  }

  __device__ uint64_t set_index(int index) {
    return atomicOr((unsigned long long int *)this, SET_BIT_MASK(index));
  }

  __device__ uint64_t unset_index(int index) {
    return atomicAnd((unsigned long long int *)this, ~SET_BIT_MASK(index));
  }

  // TODO pragmatize this
  __device__ bool unset_bit_atomic(int index) {
    uint64_t unset_mask = generate_unset_mask(index);

    uint64_t old = atomicAnd((unsigned long long int *)this, unset_mask);

    return (old & ~unset_mask);
  }

  __device__ uint64_t_bitarr global_load_this() {
    return (uint64_t_bitarr)poggers::utils::ldca((uint64_t *)this);
  }

  __device__ int get_fill() { return __popcll(bits); }

  __device__ uint64_t_bitarr swap_to_empty() {
    return (uint64_t_bitarr)atomicExch((unsigned long long int *)this, 0ULL);
  }

  __device__ bool try_swap_empty() {
    return atomicCAS((unsigned long long int *)this, ~0ULL, 0ULL) == (~0ULL);
  }

  __device__ void swap_full() {
    atomicExch((unsigned long long int *)this, ~0ULL);
  }

  __device__ bool set_bits(uint64_t ext_bits) {
    return (atomicCAS((unsigned long long int *)this, 0ULL,
                      (unsigned long long int)ext_bits) == 0ULL);
  }

  __device__ uint64_t swap_bits(uint64_t ext_bits) {
    return (atomicExch((unsigned long long int *)this, ext_bits));
    //, (unsigned long long int) ext_bits));
  }

  __device__ bool unset_bits(uint64_t ext_bits) {
    return (atomicCAS((unsigned long long int *)this,
                      (unsigned long long int)ext_bits, 0ULL) == ext_bits);
  }

  __device__ uint64_t get_bits() { return bits; }

  __device__ uint64_t set_OR_mask(uint64_t ext_bits) {
    return atomicOr((unsigned long long int *)this, ext_bits);
  }

  __device__ int bits_before_index(int index) {
    if (index == 63) {
      return __popcll(bits);
    }

    uint64_t mask = (1ULL << index);

    return __popcll(bits & mask);
  }

  __device__ void apply_mask(uint64_t mask) { bits = bits & mask; }

  __host__ __device__ operator uint64_t() const { return bits; }
};

__device__ int select_unique_bit(cg::coalesced_group &active_threads,
                                 uint64_t_bitarr &active_bits) {
  int my_bit = -1;

  // if (active_threads.thread_rank() == 0){
  // 	printf("Starting up selection process with active bits %lu\n",
  // copy_active_bits.bits);
  // }

  while (true) {
    cg::coalesced_group searching_group = cg::coalesced_threads();

#if BETA_UTIL_DEBUG
    if (searching_group.thread_rank() == 0)
      printf("Start of round: %d in group, bits %llx\n", searching_group.size(),
             active_bits.bits);
#endif

    int bit = active_bits.get_random_active_bit();

    // if not available fucken crash
    if (bit == -1) {
      my_bit = -1;
      break;
    }

    uint64_t my_mask = (1ULL) << bit;

    // now scan across the masks
    uint64_t scanned_mask =
        cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

    // final thread needs to broadcast updates
    if (searching_group.thread_rank() == searching_group.size() - 1) {
      // doesn't matter as the scan only adds bits
      // not to set the mask to all bits not taken
      uint64_t final_mask = ~(scanned_mask | my_mask);

      active_bits.apply_mask(final_mask);

#if BETA_UTIL_DEBUG
      printf(
          "Team member %d/%d sees new bits as %llx, new mask %llx, popcount of "
          "bits %d\n",
          searching_group.thread_rank(), searching_group.size(), active_bits,
          scanned_mask, __popcll(active_bits.bits));
#endif
    }

    // everyone now has an updated final copy of ext bits?
    active_bits = searching_group.shfl(active_bits, searching_group.size() - 1);

    if (!(scanned_mask & my_mask)) {
      // I received an item!
      // allocation has already been marked and index is set
      // break to recoalesce for exit
      my_bit = bit;
      break;
    }
  }

  // group needs to synchronize
  // everyone needs to ballot on what the smallest version of bits is.
  int min = cg::reduce(active_threads, __popcll(active_bits), cg::less<int>());

#if BETA_UTIL_DEBUG
  printf("Min is %d, my_popcount active_bits %llx\n", min, active_bits);
#endif

  int leader = __ffs(active_threads.ballot(__popcll(active_bits) == min)) - 1;

  // group needs to synchronize
  active_bits = active_threads.shfl(active_bits, leader);

  return my_bit;
}

}  // namespace allocators

}  // namespace poggers

#endif  // GPU_BLOCK_