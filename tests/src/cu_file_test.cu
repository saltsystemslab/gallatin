/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/gallatin.cuh>

#include <gallatin/allocators/timer.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <new>


#include "cufile.h"

using namespace gallatin::allocators;

#define MAX_BUFFER_SIZE (128 * 1024UL)


int main(int argc, char** argv) {


   CUfileError_t status;
   CUfileDrvProps_t props;

   status = cuFileDriverOpen();
   if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cufile driver open error " << std::endl;
      return -1;
   }



   const char *TESTFILE;
   CUfileDescr_t cf_descr;
   CUfileHandle_t cf_handle;
   void * devptr = NULL;

   status = cuFileDriverClose();
   if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cufile driver close failed:" << std::endl;

      return -1;
   }


   cudaDeviceReset();
   return 0;

}
