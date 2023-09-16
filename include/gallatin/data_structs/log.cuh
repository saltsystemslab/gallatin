#ifndef GALLATIN_LOG
#define GALLATIN_LOG


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
//and the global allocator
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>

#include <gallatin/data_structs/custring.cuh>

#include <gallatin/data_structs/dev_host_queue.cuh>



//This is a logger for cuda! Uses a queue structure to record
// entries with unbounded length, up to the maximum device memory

//entries maintain TID, logID, and a message generated via 
namespace gallatin {

namespace data_structs {



	struct log_entry {

		uint64_t tid;
		uint64_t log_id;

		

	}


	template <typename Allocator>
	struct logger {

	};


}


}


#endif //end of queue name guard