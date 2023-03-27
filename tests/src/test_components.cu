/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
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
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>

#include <poggers/representations/12_bit_bucket.cuh>

#include <poggers/sizing/variadic_sizing.cuh>


#include <poggers/experimental/templated_funcs.cuh>

#include <stdio.h>
#include <iostream>


#include <poggers/representations/12_bit_bucket.cuh>



using bucket_type = poggers::representations::twelve_bucket<uint64_t, uint16_t, uint16_t, 16, 4>;


using insert_type = poggers::insert_schemes::single_slot_insert<uint64_t, uint64_t, 8, 8, poggers::representations::key_val_pair, 5, poggers::hashers::murmurHasher, poggers::probing_schemes::doubleHasher>;

using table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
     // poggers::representations::key_val_pair, 8>

     //using forst_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
    
using second_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, table_type>;

using p2_table = poggers::tables::static_table<uint64_t,uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 3, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


using counter_type = poggers::experimental::example_counter;



#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct large_uint { 

   uint64_t internal_rep [8];

};

__host__ __device__ bool operator==(large_uint A, large_uint B){

   for (int i =0; i < 8; i++){
      if (A.internal_rep[i] != B.internal_rep[i]) return false;
   }

   return true;

}

__global__ void test_p2(p2_table * table){

   auto insert_tile = table->get_my_tile();

   if (insert_tile.meta_group_rank() != 0) return;

   if (insert_tile.thread_rank() == 0){
      printf("Starting test on GPU\n");
   }

   table->insert(insert_tile, 1, 1);

   uint64_t found_val = 0;

   assert(table->query(insert_tile, 1, found_val));

   assert(found_val == 1);


   table->insert(insert_tile, 2, 2);

   assert(table->query(insert_tile, 2, found_val));

   assert(found_val == 2);

   table->insert(insert_tile, 3, 4);

   assert(table->query(insert_tile, 3, found_val));

   assert(found_val == 4);

   // table->insert(insert_tile, 2, 0);

   // table->insert(insert_tile, 2, 0);

   // table->insert(insert_tile, 2, 0);

   if (insert_tile.thread_rank() == 0)printf("All tests done!\n");

}


__global__ void test_p2_nitems(p2_table * table, uint64_t nitems){

   auto insert_tile = table->get_my_tile();

   uint64_t tid = insert_tile.meta_group_size()*blockIdx.x + insert_tile.meta_group_rank();

   if (tid >= nitems) return;

   if (!table->insert(insert_tile, tid, tid)){

      if (insert_tile.thread_rank() == 0) printf("Tid %llu failed.\n", tid);

   }

   uint64_t found_val = 0;

   assert(table->query(insert_tile, tid, found_val));

   assert(found_val == tid);

}


__global__ void test_p2_nitems_query(p2_table * table, uint64_t nitems){

   auto insert_tile = table->get_my_tile();

   uint64_t tid = insert_tile.meta_group_size()*blockIdx.x + insert_tile.meta_group_rank();

   if (tid >= nitems) return;


   uint64_t found_val = 0;

   assert(table->query(insert_tile, tid, found_val));

   assert(found_val == tid);

}


__host__ void test_p2(){

   poggers::sizing::size_in_num_slots<1> first_size(1100);

   p2_table * table = p2_table::generate_on_device(&first_size, 1);


   test_p2<<<1,32>>>(table);

   p2_table::free_on_device(table);

   for (int i =1; i < 11; i++){

      poggers::sizing::size_in_num_slots<1> var_size(i);

      table = p2_table::generate_on_device(&var_size, 1);

      test_p2_nitems<<<table->get_num_blocks(i), table->get_block_size(i)>>>(table, i);

      test_p2_nitems_query<<<table->get_num_blocks(i), table->get_block_size(i)>>>(table, i);

      cudaDeviceSynchronize();

      p2_table::free_on_device(table);

      cudaDeviceSynchronize();

      printf("Done with %d\n\n\n", i);

   }

   cudaDeviceSynchronize();

   return;


}

__global__ void bucket_tests(bucket_type * bucket_arr){

      auto thread_block = cg::this_thread_block();

      cg::thread_block_tile<4> insert_tile = cg::tiled_partition<4>(thread_block);

      if (insert_tile.meta_group_rank() != 0) return;


      for (uint64_t i =0; i < 16; i++){

         assert(bucket_arr[0].insert(insert_tile, i+1, 0ULL));

      }

      assert (!bucket_arr[0].insert(insert_tile, 17ULL, 0ULL));
    

      for (uint64_t i=0; i<16;i++){
         uint16_t temp_val;
         assert(bucket_arr[0].query(insert_tile, i+1, &temp_val));

      }

      uint16_t temp_val;
      assert(!bucket_arr[0].query(insert_tile, 17, &temp_val));
     
      return;

}

// __global__ void test_key_val_pair(){


//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid != 0) return;

//    printf("Thread %llu starting!\n", tid);

//    __shared__ key_val_pair<uint64_t, uint64_t> tests [3];


//     printf("should be 1: %d\n", tests[0].atomic_swap(34, 0));


//     printf("should be 0: %d\n", tests[0].atomic_swap(34, 0));




//    large_uint large_uint_test;

//    large_uint_test.internal_rep[0] =1234;

//    __shared__ key_val_pair<large_uint, uint64_t> test2 [1] ;

//    //printf("%d\n", test2[0].atomic_swap(large_uint_test, 0));

//     if (test2[0].atomic_swap(large_uint_test, 0UL)){

//       printf("first %llu\n", test2[0].key.internal_rep[0]);
//    } else {
//       printf("Failure on swap\n");
//    }

// }


__global__ void test_insert_scheme(insert_type * insert_scheme){

   //printf("Running tests\n");

   const int tile_size = 8;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   if (insert_tile.meta_group_rank() != 0) return;

   //printf("I made it!\n");

   for (int i =50; i< 80; i++){

      //printf("i: %d\n", i);

      if (insert_scheme->insert(insert_tile, i, i+50)){

         uint64_t val = 0;
         assert(insert_scheme->query(insert_tile, i, val));
         //insert_scheme->query(insert_tile, i, val);

         //assert(val == i+50)

         //printf("%d, %llu\n", i, val);
         //printf("%d queried %llu\n", i, val);
         assert(val == i+50);

         //get rid of that stupid warning
         val = val+1;
      }

   }




}

__global__ void test_counters(counter_type * counter){

   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid !=0 ) return;

   printf("Counter start: %u\n", counter[0].counter);

   counter->ext_atomicInc(5);

   printf("Counter after Add: %u\n", counter[0].counter);

   counter->ext_atomicDec(4);

   printf("Counter after Dec: %u\n", counter[0].counter);


   poggers::experimental::Call<&counter_type::dud>(counter);

   poggers::experimental::Int_call<&counter_type::ext_atomicInc>(counter, 5UL);

   printf("Counter after experimental add: %u\n", counter[0].counter);

}


__global__ void test_table(table_type * dev_table){

   const int tile_size = 8;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = dev_table->get_my_tile();

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x + insert_tile.meta_group_rank();

   if (my_id > 100000) return;


   uint64_t key = my_id*39;
   uint64_t val = key*44;
   uint64_t found_val = 0;
   assert(dev_table->insert(insert_tile, key,val));

   assert(dev_table->query(insert_tile, key, found_val));

   assert(found_val == val);


}

__global__ void test_second_table(second_tier_table_type * dev_table){

   const int tile_size = 8;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x + insert_tile.meta_group_rank();

   if (my_id > 1500) return;


   uint64_t key = my_id*39;
   uint64_t val = key*44;
   uint64_t found_val = 0;
   assert(dev_table->insert(insert_tile, key,val));

   assert(dev_table->query(insert_tile, key, found_val));

   assert(found_val == val);


}


__global__ void offset_test(uint16_t * offset_test_array){


   const int tile_size = 1;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x+insert_tile.meta_group_rank();

   if (my_id !=0) return;

   printf("Address of main array %p\n", offset_test_array);


   //insert the first key

   //poggers::helpers::sub_byte_atomic_write<uint16_t, uint64_t, 4>(offset_test_array, 1ULL, 0, 12);

   //poggers::helpers::sub_byte_atomic_write<uint16_t, uint64_t, 4>(offset_test_array, 1ULL, 1, 12);

   poggers::helpers::sub_byte_atomic_write<uint16_t, uint64_t, 4>(offset_test_array, 1ULL, 3, 12);

   return;


}

__global__ void test_block(bucket_type * bucket){


   const int tile_size = 4;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x+insert_tile.meta_group_rank();

   if (my_id !=0) return;

   poggers::hashers::murmurHasher<uint64_t, 1> my_hasher;

   //blazin
   my_hasher.init(420);

   bucket->full_reset(insert_tile);


   for (uint64_t i = 0; i < 10; i++){

      for (uint64_t j = 0; j < 16; j++){

         uint64_t key = my_hasher.hash(i+j*100001);

         if (!bucket->insert(insert_tile, key, 0)){

            //printf("Insert Problem: %llu, %llu\n", i, j);
            //bucket->insert(insert_tile, key, 0);
         } else {

            uint16_t ext_val = 0;
            if (!bucket->query(insert_tile, key, ext_val)){

               printf("Query Problem: %llu, %llu\n", i, j);
               bucket->query(insert_tile, key, ext_val);

            }

         }

      }


      assert(!bucked->insert(insert_tile, 10, 0));

      bucket->full_reset(insert_tile);


   }



}

__global__ void test_block_grid(bucket_type * bucket){


   const int tile_size = 4;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x+insert_tile.meta_group_rank();

   if (my_id >= 16) return;

   //auto g = cg::this_grid();

   poggers::hashers::murmurHasher<uint64_t, 1> my_hasher;

   //blazin
   my_hasher.init(420);



   bucket->full_reset(insert_tile);

   //g.sync();

   __syncthreads();

   for (uint64_t i = 0; i < 10; i++){

      uint64_t key = my_hasher.hash(i*100001+my_id);

      if (!bucket->insert(insert_tile, key, 0)){

         printf("Grid Insert Problem: %llu, %llu\n", i, my_id);
         bucket->insert(insert_tile, key, 0);
      } else {

         uint16_t ext_val = 0;
         if (!bucket->query(insert_tile, key, ext_val)){

            printf("Grid Query Problem: %llu, %llu\n", i, my_id);
            bucket->query(insert_tile, key, ext_val);

         }

      }


      __syncthreads();

      assert(!bucked->insert(insert_tile, 0, 0));

      __syncthreads();

      bucket->full_reset(insert_tile);

      __syncthreads();

   }



}


__global__ void test_block_secondary(bucket_type * bucket){


   const int tile_size = 4;

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> insert_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t my_id = insert_tile.meta_group_size()*blockIdx.x+insert_tile.meta_group_rank();

   if (my_id !=0) return;

   poggers::hashers::murmurHasher<uint64_t, 1> my_hasher;

   //blazin
   my_hasher.init(42);

   bucket->full_reset(insert_tile);

   for (uint64_t i = 0; i < 10; i++){

      for (uint64_t j = 0; j < 16; j++){

         uint64_t key = my_hasher.hash(i+j*100001);

         if (!bucket->insert(insert_tile, key, 0)){

            printf("Insert Problem: %llu, %llu\n", i, j);
            bucket->insert(insert_tile, key, 0);
         } else {

            uint16_t ext_val = 0;
            if (!bucket->query(insert_tile, key, ext_val)){

               printf("Second Query Problem: %llu, %llu\n", i, j);
               bucket->query(insert_tile, key, ext_val);

            }

         }

      }

      assert(!bucked->insert(insert_tile, 10, 0));


      for (uint64_t j = 0; j < 16; j++){

         uint64_t key = my_hasher.hash(i+j*100001);

         uint16_t ext_val = 0;

         if (!bucket->query(insert_tile, key, ext_val)){

               printf("Second Seconed Query Problem: %llu, %llu\n", i, j);
               bucket->query(insert_tile, key, ext_val);

         }



      }

      bucket->full_reset(insert_tile);


   }



}



int main(int argc, char** argv) {

    // using hash_type = poggers::hashers::murmurHasher<uint64_t, 1>;
	
    //  hash_type x;
    //  x.init(64);

    //  poggers::probing_schemes::doubleHasher<uint64_t, 1, poggers::hashers::murmurHasher, 5>  prober (64);

    //  uint64_t key = 5;

    //  for (uint64_t i = prober.begin(key); i != prober.end(); i = prober.next()){

    //     printf("%llu\n", i);
    //  }

    //  printf("\n\n");

    //  poggers::probing_schemes::powerOfTwoHasher<uint64_t, 1, poggers::hashers::murmurHasher, 2> p2_prober(53);

    //  for (uint64_t i = p2_prober.begin(key); i != p2_prober.end(); i = p2_prober.next()){

    //     printf("%llu\n", i);
    //  }


    //  insert_type test;

    //  insert_type * alt_test = insert_type::generate_on_device(50,42);
     
     
    

    //   test_insert_scheme<<<1,32>>>(alt_test);

    //   cudaDeviceSynchronize();

    //   insert_type::free_on_device(alt_test);



    //  //test_key_val_pair<<<1,1>>>();

    //  gpuErrorCheck(cudaPeekAtLastError());
    //  gpuErrorCheck(cudaDeviceSynchronize());


    //  printf("Starting Tests for Init scheme\n");


    //  poggers::sizing::size_in_num_slots<1> first_size(1000000);

    //  poggers::sizing::size_in_num_slots<4> second_size((uint64_t) 14, (uint64_t) 52, 12341234UL, 1000000000000UL);


    //  for (int i =0; i < 4; i++){

    //      printf("%llu\n", second_size.next());

    //  }
    //  assert(second_size.next() == second_size.end());


    //  //test table init

    //  //poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, poggers::insert_schemes::single_slot_insert, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, false, void>  my_table;

    
    //  table_type table;


    //  table_type * dev_table = table_type::generate_on_device(&first_size, 15);

    //  cudaDeviceSynchronize();

    //  test_table<<<100000, 1024>>>(dev_table);

    //  cudaDeviceSynchronize();


    //  table_type::free_on_device(dev_table);

    //  poggers::sizing::size_in_num_slots<2> alt_table_sizing(1000,1000);

    //  second_tier_table_type * alt_dev_table = second_tier_table_type::generate_on_device(&alt_table_sizing, 420);

    //  cudaDeviceSynchronize();

    //  test_second_table<<<10000, 1024>>>(alt_dev_table);

    //  cudaDeviceSynchronize();

    //  second_tier_table_type::free_on_device(alt_dev_table);

    //  cudaDeviceSynchronize();


    //  //power_of_two_tests

    //  poggers::insert_schemes::power_of_n_insert_scheme<uint64_t,uint64_t, 8, 8, poggers::representations::key_val_pair, 2, poggers::hashers::murmurHasher, poggers::probing_schemes::doubleHasher> powerNHasher;

    //  test_p2();

    //  cudaDeviceSynchronize();

    //  using test_key_type = poggers::representations::shortened_bitmask_key_val_pair<uint64_t,uint64_t,uint16_t>;

    //  test_key_type test_key;

    //  assert(test_key.is_empty());

    //  test_key_type alt_test_key(15ULL, 16ULL);

    //  assert(!alt_test_key.is_empty());

    //  assert(alt_test_key.contains(15ULL));


    //  using small_key_type = poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair<uint64_t,uint64_t>;


    //  small_key_type final_test(15ULL, 15ULL);

    //   assert(!final_test.is_empty());

    //  assert(final_test.contains(15ULL));


    //  using shortened_table = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

    //  poggers::sizing::size_in_num_slots<1> shortened_sizer(100);
    //  shortened_table * my_table;

    //  my_table = shortened_table::generate_on_device(&shortened_sizer, 123);

    //  shortened_table::free_on_device(my_table);


    //  poggers::sizing::variadic_size var_size(1,2);

    //  printf("%llu\n", var_size.next());
    //  printf("%llu\n", var_size.next());
    //  printf("%llu\n", var_size.next());

    //  poggers::sizing::variadic_size alt_var_size(1,2000);

    //  second_tier_table_type * variadic_table = second_tier_table_type::generate_on_device(&alt_var_size, 15);

    //  second_tier_table_type::free_on_device(variadic_table);

    //  cudaDeviceSynchronize();



    //  counter_type * counters;

    //  cudaMalloc((void **)& counters, sizeof(counter_type)*1);

    //  cudaDeviceSynchronize();

    //  //test_counters<<<1,1>>>(counters);

    //  cudaDeviceSynchronize();

    //  cudaFree(counters);


     printf("Starting half byte tests\n");


     bucket_type my_bucket;

     bucket_type * bucket_arr;

     cudaMalloc((void **)&bucket_arr, sizeof(bucket_type));

     test_block<<<1,4>>>(bucket_arr);

     // bucket_tests<<<1,4>>>(bucket_arr);

     cudaDeviceSynchronize();

     test_block_secondary<<<1,4>>>(bucket_arr);

     cudaDeviceSynchronize();

     test_block_grid<<<1, 1024>>>(bucket_arr);

     cudaDeviceSynchronize();

     cudaFree(bucket_arr);


     uint16_t * offset_test_array;

     cudaMalloc((void**)&offset_test_array, sizeof(uint16_t)*3);

     //init done

     printf("Address from the outside: %p\n", offset_test_array);

     offset_test<<<1,1>>>(offset_test_array);

     cudaDeviceSynchronize();



     cudaFree(offset_test_array);


     printf("All tests succeeded\n");

	return 0;

}
