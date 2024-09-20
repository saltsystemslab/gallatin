/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/alloc_utils.cuh>

#include <gallatin/allocators/timer.cuh>


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <new>
#include <cstdint>


__global__ void ptr_hop_kernel(uint64_t * ptrs, uint64_t n_pointers){


   uint64_t next_arg = 0;

   uint64_t n_traversed = 0;


   do {

      //printf("Traversing %lu\n", next_arg);

      uint64_t * next = ptrs+next_arg;

      next_arg = next[0];

      n_traversed+=1;

   } while (next_arg != 0);

   if (n_traversed != n_pointers){
      printf("Traversed %lu\n", n_traversed);
   }


}


__host__ void host_hop(uint64_t * ptrs, uint64_t n_pointers){


   gallatin::utils::timer host_timing;

   uint64_t next_arg = 0;

   do {

      uint64_t * next = ptrs+next_arg;

      next_arg = next[0];

   } while (next_arg != 0);


   host_timing.sync_end();

   host_timing.print_throughput("Host Traversed", n_pointers);


}


__host__ void device_hop(uint64_t * ptrs, uint64_t n_pointers){



   uint64_t * device_version;

   cudaMalloc((void **)&device_version, sizeof(uint64_t)*n_pointers);

   cudaMemcpy(device_version, ptrs, sizeof(uint64_t)*n_pointers, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer device_timing;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_timing.sync_end();

   device_timing.print_throughput("Device Traversed", n_pointers);

   gallatin::utils::timer device_timing_2;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_timing_2.sync_end();

   device_timing_2.print_throughput("Device Traversed (warm)", n_pointers);

   cudaFree(device_version);


}


__host__ void device_host_hop(uint64_t * ptrs, uint64_t n_pointers){



   uint64_t * device_version;

   cudaMallocHost((void **)&device_version, sizeof(uint64_t)*n_pointers);

   memcpy(device_version, ptrs, sizeof(uint64_t)*n_pointers);

   cudaDeviceSynchronize();

   gallatin::utils::timer device_timing;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_timing.sync_end();

   device_timing.print_throughput("Device Traversed Host", n_pointers);

   gallatin::utils::timer device_timing_2;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_timing_2.sync_end();

   device_timing_2.print_throughput("Device Traversed Host (Warm)", n_pointers);

   cudaFreeHost(device_version);


}

__host__ void device_managed_hop_host_resident(uint64_t * ptrs, uint64_t n_pointers){



   uint64_t * device_version;

   cudaMallocManaged((void **)&device_version, sizeof(uint64_t)*n_pointers);

   cudaMemcpy(device_version, ptrs, sizeof(uint64_t)*n_pointers, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   host_hop(device_version, n_pointers);

   gallatin::utils::timer device_repeat_timing;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_repeat_timing.sync_end();

   device_repeat_timing.print_throughput("Managed Traversed (Warm)", n_pointers);


   cudaFree(device_version);


}


__host__ void device_managed_hop(uint64_t * ptrs, uint64_t n_pointers){



   uint64_t * device_version;

   cudaMallocManaged((void **)&device_version, sizeof(uint64_t)*n_pointers);

   cudaMemcpy(device_version, ptrs, sizeof(uint64_t)*n_pointers, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer device_timing;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_timing.sync_end();

   device_timing.print_throughput("Managed Traversed", n_pointers);


   gallatin::utils::timer device_repeat_timing;

   ptr_hop_kernel<<<1,1>>>(device_version, n_pointers);

   device_repeat_timing.sync_end();

   device_repeat_timing.print_throughput("Managed Traversed (Warm)", n_pointers);


   cudaFree(device_version);


}



//generate a shuffled list for the ptr hop.
//the list must touch every item once, and touch no items twice.
__host__ uint64_t * generate_shuffled_data(uint64_t n_items, uint64_t stride){


   printf("Loading %llu bytes\n", sizeof(uint64_t)*nitems);

   uint64_t * data = (uint64_t *) malloc(sizeof(uint64_t)*n_items);


   for (uint64_t i=0; i < n_items; i++){

      uint64_t next = i+stride;

      if (next >= n_items) next = (next) % n_items;

      data[i] = next;

   }

   printf("Data generated\n");

   return data;

}




int main(int argc, char** argv) {

   uint64_t n_hops;

   uint64_t stride;


   if (argc < 2){
      n_hops = 4096;
   } else {
      n_hops = std::stoull(argv[1]);
   }

   if (argc < 3){
      stride = 128;
   } else {
      stride = std::stoull(argv[2]);
   }



   uint64_t * data_ptrs = generate_shuffled_data(n_hops, stride);


   host_hop(data_ptrs, n_hops);

   //device_hop(data_ptrs, n_hops);
   
   device_managed_hop(data_ptrs, n_hops);

   device_host_hop(data_ptrs, n_hops);

   device_managed_hop_host_resident(data_ptrs, n_hops);

   free(data_ptrs);


   cudaDeviceReset();
   return 0;

}
