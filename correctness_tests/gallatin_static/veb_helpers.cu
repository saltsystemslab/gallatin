#ifndef VEB_HELPER_CU
#define VEB_HELPER_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define GALLATIN_DEBUG 1


#include <veb_helpers.hpp>
#include <gallatin/allocators/veb_components.cuh>
#include <gallatin/allocators/veb.cuh>


using namespace gallatin::internals;


//using internal_bitarry_type = internal_bitarray;

//kernels

template <uint size>
__global__ void testBitarrayInitFFSKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  if (bits[0].ffs() != 0){

    printf("Size %u failed\n", size);
    write_global_log(4);
  }

}


template <uint size>
__global__ void testBitarrayAtomicsKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  for (uint i =0; i < internal_bitarray<size>::n_bits; i++){

    if (!bits[0].set_loc_atomic(i)){
      write_global_log(4, i, 0);
    }

    if (bits[0].set_loc_atomic(i)){
      write_global_log(4, i, 1);
    }

  }

}

template <uint size>
__global__ void testBitarrayFfsAtomicKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  for (uint i = internal_bitarray<size>::n_bits; i > 0; i--){

    uint index = i-1;

    if (!bits[0].set_loc_atomic(index)){
      write_global_log(4, index, 0);
    }

    if (bits[0].ffs()-1 != index){
      write_global_log(4, index, 1);
    }

  }

}


template <uint size>
__global__ void testAtomicsSetUnsetKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;


  for (uint i = 0; i < internal_bitarray<size>::n_bits; i++){

    if (!bits[0].set_loc_atomic(i)){
      write_global_log(4);
    }

    if (bits[0].ffs()-1 != i){
      write_global_log(4);
    }

    if (!bits[0].unset_loc_atomic(i)){
      write_global_log(4);
    }

    if (bits[0].ffs()-1 != -1){
      write_global_log(4);
    }

  }

}


template <uint size>
__global__ void testBitarrayExceptionsKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //write two invalid locations.
  bits[0].set_loc_atomic(-1);

  bits[0].set_loc_atomic(internal_bitarray<size>::n_bits);

}


template <uint size>
__global__ void testLdAcqKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //write two invalid locations.
  bits[0].set_loc_atomic(1);

  internal_bitarray<size> loaded_copy = bits[0].ld_acq();

  if (loaded_copy.ffs()-1 != 1){
    write_global_log(4);
  }

}


template <uint size>
__global__ void testGroupSetKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //write two invalid locations.

  if (!bits[0].set_contiguous(internal_bitarray<size>::n_bits, 0)){

    write_global_log(4, 0);
    return;
  }

  if (!bits[0].popc() == internal_bitarray<size>::n_bits){
    write_global_log(4, 1);
    return;
  }

  //printf("Bits: %lx\n", bits[0].data);

  if (!bits[0].unset_contiguous(internal_bitarray<size>::n_bits, 0)){
    write_global_log(4, 2);
    return;
  }


  uint half_size = internal_bitarray<size>::n_bits/2;


  if (!bits[0].set_contiguous(internal_bitarray<size>::n_bits, 0)){
    write_global_log(4, 3);
    return;
  }


  if (!bits[0].unset_contiguous(half_size, 0)){
    write_global_log(4, 4);
    return;
  }

  if (!bits[0].popc() == half_size){
    write_global_log(4, 5);
    return;
  }


  if (!bits[0].unset_contiguous(half_size, half_size)){
    write_global_log(4, 6);
    return;
  }

  if (!bits[0].popc() == 0){
    write_global_log(4, 7);
    return;
  }

  //test off strides

  for (int i = 0; i < half_size; i++){

    if (!bits[0].set_contiguous(half_size, i)){
      write_global_log(4, i, 0);
      return;
    }

    if (!bits[0].popc() == half_size){
      write_global_log(4, i, 1);
      return;
    }


    if (!bits[0].unset_contiguous(half_size, i)){
      write_global_log(4, i, 2);
      return;
    }

    if (!(bits[0].popc() == 0)){
      write_global_log(4, i, 3);
      return;
    }

  }


  //test failures.

  if (!bits[0].set_contiguous(1, 10)){
      write_global_log(4, 8);
      return;
  }

  if (!(bits[0].popc() == 1)){
      write_global_log(4, 9);
      return;
  }

  if (bits[0].set_contiguous(15, 0)){

      write_global_log(4, 10);
      return;
  }

  if (bits[0].unset_contiguous(15, 0)){
      write_global_log(4, 11);
      return;
  }

  if (!bits[0].unset_loc_atomic(15)){
      write_global_log(4, 12);
      return;
  }


}


template <uint size>
__global__ void testClaimFirstKernel(internal_bitarray<size> * bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //write two invalid locations.

  bits[0].set_contiguous(internal_bitarray<size>::n_bits, 0);

  if (bits[0].popc() != internal_bitarray<size>::n_bits){
    write_global_log(4, 0);
    return;
  }

  internal_bitarray<size> copy_bits;

  for (uint i = 0; i < internal_bitarray<size>::n_bits; i++){

    if (bits[0].claim_first(copy_bits) != i){
      write_global_log(4, i, 1);
      return;
    }

    if (bits[0].popc() != internal_bitarray<size>::n_bits-1-i){
      write_global_log(4, i, 2);
      return;
    }

    if (bits[0].ffs()-1 != i+1 && i != internal_bitarray<size>::n_bits-1){

      write_global_log(4, i, 3);
      return;
    }


    if (i == internal_bitarray<size>::n_bits-1 && bits[0].ffs() != 0){

      write_global_log(4, i, 4);
      return;
    }

  }

  if (bits[0].claim_first(copy_bits) != -1){
    write_global_log(4, internal_bitarray<size>::n_bits+1);
  }

}


template <uint size>
__global__ void testVebBasicKernel(veb<size> * tree, uint n_bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;


  for (uint i = 0; i < n_bits; i++){

    if (!tree->remove(i)){
      write_global_log(4, 0, i);
      return;
    }

  }

  for (uint i = 0; i < n_bits; i++){

    if (tree->query(i)){
      write_global_log(4, 1, i);
      return;
    }

  }

  for (uint i = 0; i < n_bits; i++){

    if (!tree->insert(i)){
      write_global_log(4, 2, i);
      return;
    }

  }

  for (uint i = 0; i < n_bits; i++){

    if (!tree->query(i)){
      write_global_log(4, 3, i);
      return;
    }

  }

}


template <uint size>
__global__ void testFindFirstKernel(veb<size> * tree, uint n_bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //first clear
  for (uint i = 0; i < n_bits; i++){

    if (!tree->remove(i)){
      write_global_log(4, 0, i);
      return;
    }

  }

  for (uint i = 0; i < n_bits; i++){

    if (!tree->insert(i)){
      write_global_log(4, 2, i);
      return;
    }


    if (tree->find_first(0) != i){
      write_global_log(4, 3, i);
      return;
    }


    if (!tree->remove(i)){
      write_global_log(4, 4, i);
      return;
    }

  }



}

template <uint size>
__global__ void testClaimFirstKernel(veb<size> * tree, uint n_bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  //first clear

  for (uint i = 0; i < n_bits; i++){

    if(tree->claim_first(0) != i){
      write_global_log(4, 0, i);
    }

  }

  if (tree->claim_first(0) != veb<size>::fail()){
    write_global_log(4, 1, 0);
  }

  for (uint i = 0; i < n_bits; i++){
    if (!tree->insert(i)){
        write_global_log(4, 2, i);
        return;
    }
  }


  for (uint i = n_bits; i > 1; i--){

    uint index = i-1;

    if(tree->claim_first(index) != index){
      write_global_log(4, 3, i);
    }

  }

 

}

template <uint size>
__global__ void testParallelKernel(veb<size> * tree, uint64_t * bits, uint n_bits){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_bits) return;

  //first clear


  uint tree_claim = tree->claim_first(0);

  if (tree_claim == veb<size>::fail()){
    write_global_log(4, 0, tid);
  }
 

  uint64_t high = tree_claim/veb<size>::component_type::n_bits;
  uint64_t low = tree_claim % veb<size>::component_type::n_bits;


  uint64_t previous = atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low));

  if (previous & SET_BIT_MASK(low)){
    write_global_log(4, 1, tree_claim);
  }


}

//end of kernels



template<uint size>
internal_bitarray<size> * setupBitarray(uint64_t n_copies){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = gallatin::utils::get_device_version<bitarr_type>(n_copies);

  cudaMemset(dev_version, 0, sizeof(bitarr_type)*n_copies);

  return dev_version;


}

template<uint size>
void teardownBitarray(internal_bitarray<size> * dev_version){
  cudaFree(dev_version);
}

template <uint size> 
void testInitFFSOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testBitarrayInitFFSKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testAtomicsOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testBitarrayAtomicsKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}


template <uint size> 
void testFfsAtomicOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testBitarrayFfsAtomicKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testBitarrExceptionsOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testBitarrayExceptionsKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testAtomicsSetUnsetOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testAtomicsSetUnsetKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testLdAcqOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testLdAcqKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testGroupSetOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testGroupSetKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}

template <uint size> 
void testClaimFirstOneSize(){

  using bitarr_type = internal_bitarray<size>;

  bitarr_type * dev_version = setupBitarray<size>(1);

  testClaimFirstKernel<size><<<1,1>>>(dev_version);

  GPUErrorCheck(cudaDeviceSynchronize());

  teardownBitarray<size>(dev_version);

}


template <uint size> 
void testVebBasicOpsOneSize(uint64_t n_bits){


  veb<size> * test_tree = veb<size>::generate_on_device(n_bits);

  testVebBasicKernel<size><<<1,1>>>(test_tree, n_bits);

  GPUErrorCheck(cudaDeviceSynchronize());

  veb<size>::free_on_device(test_tree);

  return;

}


template <uint size> 
void testVebFindFirstOneSize(uint64_t n_bits){


  veb<size> * test_tree = veb<size>::generate_on_device(n_bits);

  testFindFirstKernel<size><<<1,1>>>(test_tree, n_bits);

  GPUErrorCheck(cudaDeviceSynchronize());

  veb<size>::free_on_device(test_tree);

  return;

}




template <uint size> 
void testVebFindClaimFirstOneSize(uint64_t n_bits){


  veb<size> * test_tree = veb<size>::generate_on_device(n_bits);

  testClaimFirstKernel<size><<<1,1>>>(test_tree, n_bits);

  GPUErrorCheck(cudaDeviceSynchronize());

  veb<size>::free_on_device(test_tree);

  return;

}

template <uint size> 
void testVebFindParallelOneSize(uint64_t n_bits){

  uint64_t n_uints = (n_bits-1)/64+1;

  uint64_t * bits = gallatin::utils::get_device_version<uint64_t>(n_uints);

  cudaMemset(bits, 0, sizeof(uint64_t)*n_uints);

  veb<size> * test_tree = veb<size>::generate_on_device(n_bits);

  testParallelKernel<size><<<(n_bits-1)/256+1,256>>>(test_tree, bits, n_bits);

  GPUErrorCheck(cudaDeviceSynchronize());

  veb<size>::free_on_device(test_tree);

  cudaFree(bits);

  return;

}


//veb helper functions


template <uint size>
bool assert_veb_size(uint64_t n_bits, uint64_t n_levels){

  veb<size> * test_tree = veb<size>::generate_on_device(n_bits);


  veb<size> * host_copy = gallatin::utils::copy_to_host<veb<size>>(test_tree);

  bool result = host_copy->n_levels == n_levels;

  cudaFreeHost(host_copy);

  veb<size>::free_on_device(test_tree);

  return result;

}


void veb_helper_tests::open_global_log(){
  gallatin::internals::init_global_error_log();
}

int veb_helper_tests::close_global_log(){
  return gallatin::internals::close_global_error_log();
}

bool veb_helper_tests::testComponentSizes(){


  static_assert(sizeof(internal_bitarray<4>) == 4);

  static_assert(sizeof(internal_bitarray<8>) == 8);
  static_assert(sizeof(internal_bitarray<16>) == 16);



  return true;

}

bool veb_helper_tests::testInitFFS(){

  open_global_log();

  testInitFFSOneSize<4>();
  testInitFFSOneSize<8>();
  testInitFFSOneSize<16>();



  return (close_global_log() == 0);

}

bool veb_helper_tests::testAtomics(){

  open_global_log();

  testAtomicsOneSize<4>();
  testAtomicsOneSize<8>();
  testAtomicsOneSize<16>();



  return (close_global_log() == 0);

}


bool veb_helper_tests::testFfsAtomic(){

  open_global_log();

  testFfsAtomicOneSize<4>();
  testFfsAtomicOneSize<8>();
  testFfsAtomicOneSize<16>();



  return (close_global_log() == 0);

}



bool veb_helper_tests::testExcepts(){

  open_global_log();

  testBitarrExceptionsOneSize<4>();
  testBitarrExceptionsOneSize<8>();
  testBitarrExceptionsOneSize<16>();


  int n_excepts = close_global_log();

  if (n_excepts != 6){
    printf("n_excepts %d\n", n_excepts);
  }

  return (n_excepts == 6);

}


bool veb_helper_tests::testSetUnset(){

  open_global_log();

  testAtomicsSetUnsetOneSize<4>();
  testAtomicsSetUnsetOneSize<8>();
  testAtomicsSetUnsetOneSize<16>();



  return (close_global_log() == 0);

}


bool veb_helper_tests::testLdAcq(){

  open_global_log();

  testLdAcqOneSize<4>();
  testLdAcqOneSize<8>();
  testLdAcqOneSize<16>();



  return (close_global_log() == 0);

}


bool veb_helper_tests::testGroupSet(){

  open_global_log();

  testGroupSetOneSize<4>();
  testGroupSetOneSize<8>();
  testGroupSetOneSize<16>();



  return (close_global_log() == 0);

}


bool veb_helper_tests::testClaimFirst(){

  open_global_log();

  testClaimFirstOneSize<4>();
  testClaimFirstOneSize<8>();
  testClaimFirstOneSize<16>();



  return (close_global_log() == 0);


}


bool veb_helper_tests::testVebInit(){


  if (!assert_veb_size<4>(32, 1)){
    return false;
  }

  if (!assert_veb_size<4>(33, 2)){
    return false;
  }

  if (!assert_veb_size<4>(1024, 2)){
    return false;
  }

  if (!assert_veb_size<4>(1025, 3)){
    return false;
  }


  if (!assert_veb_size<8>(64, 1)){
    return false;
  }

  if (!assert_veb_size<8>(65, 2)){
    return false;
  }

  if (!assert_veb_size<8>(4096, 2)){
    return false;
  }

  if (!assert_veb_size<8>(4097, 3)){
    return false;
  }

  if (!assert_veb_size<16>(128, 1)){
    return false;
  }

  if (!assert_veb_size<16>(129, 2)){
    return false;
  }

  if (!assert_veb_size<16>(16384, 2)){
    return false;
  }

  if (!assert_veb_size<16>(16385, 3)){
    return false;
  }




  return true;

}

bool veb_helper_tests::testVebBasicOps(){

  open_global_log();



  testVebBasicOpsOneSize<4>(16384);
  testVebBasicOpsOneSize<8>(16384);
  testVebBasicOpsOneSize<16>(16384);



  return (close_global_log() == 0);


}


bool veb_helper_tests::testVebFindFirst(){

  open_global_log();



  testVebFindFirstOneSize<4>(16384);
  testVebFindFirstOneSize<8>(16384);
  testVebFindFirstOneSize<16>(16384);



  return (close_global_log() == 0);


}


bool veb_helper_tests::testVebClaimFirst(){

  open_global_log();



  testVebFindClaimFirstOneSize<4>(16384);
  testVebFindClaimFirstOneSize<8>(16384);
  testVebFindClaimFirstOneSize<16>(16384);



  return (close_global_log() == 0);


}


bool veb_helper_tests::testVebParallel(){

  open_global_log();


  for (int i = 0; i < 10; i++){

    testVebFindParallelOneSize<4>(16384);
    testVebFindParallelOneSize<8>(16384);
    testVebFindParallelOneSize<16>(16384);

  }





  return (close_global_log() == 0);


}




#endif




