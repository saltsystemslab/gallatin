/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

// Test for fused (device + host fallback) allocation.
// Boots a small device allocator and larger host allocator,
// then allocates 2x the device capacity using global_malloc_fused.
// All allocations should succeed by spilling to host.
// Verifies correctness and that free routes to the correct allocator.

#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::allocators;

#define TEST_BLOCK_SIZE 256


__global__ void fused_insert(uint64_t num_inserts, uint64_t size, uint64_t ** bitarray, uint64_t * misses, uint64_t * host_count){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_inserts) return;

   uint64_t * alloc = (uint64_t *) global_malloc_fused(size);

   if (alloc == nullptr){
      atomicAdd((unsigned long long int *)misses, 1ULL);
      bitarray[tid] = nullptr;
      return;
   }

   // Track how many went to host
   if (!global_gallatin->owns_allocation((void *)alloc)){
      atomicAdd((unsigned long long int *)host_count, 1ULL);
   }

   atomicExch((unsigned long long int *)&bitarray[tid], (unsigned long long int) alloc);

   alloc[0] = tid;

   __threadfence();
}


__global__ void fused_free(uint64_t num_allocs, uint64_t ** bitarray){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_allocs) return;

   uint64_t * alloc = bitarray[tid];

   if (alloc == nullptr) return;

   if (alloc[0] != tid){
      printf("Correctness error: tid %lu read %lu\n", tid, alloc[0]);
      return;
   }

   global_free_fused(alloc);

   __threadfence();
}


__host__ void run_fused_test(uint64_t device_bytes, uint64_t host_bytes, uint64_t num_allocs, int num_rounds, uint64_t size){

   uint64_t device_segments = gallatin::utils::get_max_chunks<16ULL*1024*1024>(device_bytes);

   printf("Fused test: %lu device segments, %lu allocs, size %lu\n",
          device_segments, num_allocs, size);

   gallatin::utils::timer boot_timing;

   init_global_allocator_combined(device_bytes, host_bytes, 42);

   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   uint64_t ** bits;
   cudaMalloc((void **)&bits, sizeof(uint64_t *) * num_allocs);
   cudaMemset(bits, 0, sizeof(uint64_t *) * num_allocs);

   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   uint64_t * host_count;
   cudaMallocManaged((void **)&host_count, sizeof(uint64_t));

   cudaDeviceSynchronize();

   for (int i = 0; i < num_rounds; i++){

      misses[0] = 0;
      host_count[0] = 0;

      printf("Round %d/%d\n", i, num_rounds);

      gallatin::utils::timer malloc_timing;
      fused_insert<<<(num_allocs-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(num_allocs, size, bits, misses, host_count);
      malloc_timing.sync_end();
      malloc_timing.print_throughput("Fused malloc", num_allocs);

      printf("  Misses: %lu, Host allocs: %lu / %lu\n", misses[0], host_count[0], num_allocs);

      if (misses[0] > 0){
         printf("  ERROR: %lu allocations failed - host fallback did not cover overflow\n", misses[0]);
      }

      if (host_count[0] == 0){
         printf("  WARNING: no allocations went to host - device was not saturated\n");
      }

      gallatin::utils::timer free_timing;
      fused_free<<<(num_allocs-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(num_allocs, bits);
      free_timing.sync_end();
      free_timing.print_throughput("Fused free", num_allocs);

      cudaMemset(bits, 0, sizeof(uint64_t *) * num_allocs);
      cudaDeviceSynchronize();
   }

   print_global_stats_combined();

   cudaFree(misses);
   cudaFree(host_count);
   cudaFree(bits);

   free_global_allocator_combined();
}


int main(int argc, char** argv) {

   uint64_t device_segments = 100;
   int num_rounds = 1;
   uint64_t size = 16;

   if (argc >= 2) device_segments = std::stoull(argv[1]);
   if (argc >= 3) num_rounds = std::stoull(argv[2]);
   if (argc >= 4) size = std::stoull(argv[3]);

   uint64_t device_bytes = device_segments * 16ULL * 1024 * 1024;
   // Host gets 2x device so there's room for the overflow
   uint64_t host_bytes = device_bytes * 2;

   // Compute alloc count: 2x what the base device can hold
   uint64_t mem_segment_size = 16ULL*1024*1024;
   uint64_t allocs_per_segment = mem_segment_size / size;
   uint64_t max_per_segment = mem_segment_size / 16;
   if (allocs_per_segment > max_per_segment) allocs_per_segment = max_per_segment;
   uint64_t num_allocs = allocs_per_segment * device_segments * 2;

   // First: device-only baseline (device is big enough, same alloc count, no spillover)
   printf("=== Device-only baseline (no spillover) ===\n");
   run_fused_test(device_bytes * 2, host_bytes, num_allocs, num_rounds, size);

   // Second: forced spillover to host (device is half-sized)
   printf("\n=== Fused with spillover ===\n");
   run_fused_test(device_bytes, host_bytes, num_allocs, num_rounds, size);

   cudaDeviceReset();
   return 0;
}
