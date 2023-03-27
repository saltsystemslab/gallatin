/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define DEBUG_ASSERTS 0

#define DEBUG_PRINTS 0



#include <poggers/allocators/cms.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define stack_bytes 32768

#define MEGABYTE 1024*1024

#define GIGABYTE 1024*MEGABYTE


using shibboleth = poggers::allocators::shibboleth<stack_bytes, 10, 4>;


__global__ void cms_single_threaded(shibboleth * cms){

   uint64_t test_size = 4096;


   uint ** address_array = (uint **) cms->cms_malloc(4096*sizeof(uint *));

   if (address_array == nullptr){
      printf("address_array malloc failed\n");
      asm("trap;");
   } 

   for (uint64_t i = 0; i < test_size; i++){
      address_array[i] = (uint *) cms->cms_malloc(4);

      if (address_array[i] == nullptr){
         printf("Could not allocate %llu\n", i);
         asm("trap;");
      }
   }

   for (uint64_t i = 0; i < test_size; i++){

      address_array[i][0] = i;
   }


   for (uint64_t i = 0; i < test_size; i++){

      if (address_array[i][0] != i){
         printf("Memory corrupted at %llu, shows %llu instead of %llu\n", i, address_array[i][0], i);
         asm("trap;");
      }
   }

   for (uint64_t i = 0; i< test_size; i++){
      cms->cms_free(address_array[i]);
   }


   cms->cms_free(address_array);

}


int main(int argc, char** argv) {


   //allocate 
   //const uint64_t meg = 1024*1024;
   const uint64_t bytes_in_use = 8*MEGABYTE;


   shibboleth * allocator = shibboleth::init(bytes_in_use);




   cudaDeviceSynchronize();


   cms_single_threaded<<<1,1>>>(allocator);

   cudaDeviceSynchronize();


   cms_single_threaded<<<1, 100>>>(allocator);


   shibboleth::free_cms_allocator(allocator);

   cudaDeviceSynchronize();


   return 0;




}
