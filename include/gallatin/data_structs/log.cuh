#ifndef GALLATIN_LOG
#define GALLATIN_LOG


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <poggers/allocators/alloc_utils.cuh>

//needs a queue structure to record operations
#include <poggers/data_structs/queue.cuh>


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