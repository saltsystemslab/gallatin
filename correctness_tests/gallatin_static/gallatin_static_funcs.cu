
#ifndef GAL_STATIC_FUNC_CU
#define GAL_STATIC_FUNC_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gallatin_static_funcs.hpp>
#include <gallatin/allocators/config.cuh>





void helper_open_global_log(){


  printf("Enabling global log\n");
  gallatin::internals::init_global_error_log();
}

int helper_close_global_log(){
  return gallatin::internals::close_global_error_log();
}



__global__ void log_failures_kernel(int n_failures){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_failures) return;


  gallatin::internals::write_global_log(0, tid);


}


void log_n_failures(int n_failures){

  log_failures_kernel<<<(n_failures-1)/256+1,256>>>(n_failures);

  GPUErrorCheck(cudaDeviceSynchronize());

}

#endif