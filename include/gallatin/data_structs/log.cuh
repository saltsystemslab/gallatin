#ifndef GALLATIN_LOG
#define GALLATIN_LOG


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iterator>

//alloc utils needed for easy host_device transfer
//and the global allocator
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>

#include <gallatin/data_structs/custring.cuh>

#include <gallatin/data_structs/fixed_vector.cuh>


//This is a logger for cuda! Uses a queue structure to record
// entries with unbounded length, up to the maximum device memory

//entries maintain TID, logID, and a message generated via 
namespace gallatin {

namespace data_structs {


	template <typename log_vector>
	__global__ void free_log_strings(log_vector * logs, uint64_t nitems){

		uint64_t tid = gallatin::utils::get_tid();

		if (tid >= nitems) return;

		auto log_entry = logs[0][tid];

		auto log_string = log_entry.message;

		log_string.release_string();


	};


	template <bool on_host>
	struct log_entry {

		uint64_t tid;

		custring<on_host> message;

		__device__ log_entry(custring<on_host> new_message){


			tid = gallatin::utils::get_tid();
			message = new_message;

		}

		__device__ log_entry(uint64_t ext_tid, custring<on_host> new_message){

			tid = ext_tid;
			message = new_message;

		}

		log_entry() = default;

		// template<typename ... Args>
		// __device__ log_entry(Args...all_args){

		// 	tid = gallatin::utils::get_tid();

		// 	message = make_string<on_host, Args...>(all_args...);

		// }

		//assumes log host is on host but custring is maybe not.
		__host__ std::string export_log(){


			if (on_host){

				std::string my_string(message.data());

				//std::cout << "String is " << my_string << std::endl; 
				return my_string;

			} else {

				//need buffer for host_copy
				char * host_buffer;

				//printf("Buffer has length %lu with start %lx\n", message.length, (uint64_t) message.chars);

				cudaMallocHost((void **)&host_buffer, sizeof(char)*(message.length+1));

				cudaMemcpy(host_buffer, message.data(), message.length+1, cudaMemcpyDeviceToHost);

				cudaDeviceSynchronize();

				std::string my_string(host_buffer);

				cudaFreeHost(host_buffer);

				//std::cout << "String is " << my_string << std::endl; 

				return my_string;

			}

		}



	};



	//implementation uses vector now... - embarassingly parallel and much easier to write
	template <bool on_host=false>
	struct gallatin_log {

		using my_type = gallatin_log<on_host>;

		using log_type = log_entry<on_host>;

		using custring_type = custring<on_host>;

		//10 trillion log entries
		//if you try to hit this it will break.
		using vector_type = fixed_vector<log_type, 128ULL, 100000000000ULL, on_host>;

		vector_type * storage_vector;



		static __host__ my_type * generate_on_device(){

			my_type * host_version = gallatin::utils::get_host_version<my_type>();

			host_version->storage_vector = vector_type::get_device_vector();

			return gallatin::utils::move_to_device(host_version);


		}

		static __host__ void free_on_device(my_type * device_log){


			return;
			auto host_version = gallatin::utils::move_to_host(device_log);

			//non destructive copy so that the vector components can be freed
			auto host_vector = gallatin::utils::copy_to_host(host_version->storage_vector);


			uint64_t n_logs = host_vector->size;

			free_log_strings<vector_type><<<(n_logs-1)/256+1, 256>>>(host_version->storage_vector, n_logs);

			cudaDeviceSynchronize();

			vector_type::free_device_vector(host_version->storage_vector);

			cudaFreeHost(host_vector);

			cudaFreeHost(host_version);

		}

		
		template <typename ... Args>
		__device__ void add_log(Args...all_args){


			auto string_message = make_string<on_host, Args...>(all_args...);

			log_entry new_log_entry(string_message);

			storage_vector->insert(new_log_entry);


		}


		//dump log to a host vector for easy export.
		//general steps:
		static __host__ std::vector<std::string> export_log(my_type * device_version){

			my_type * host_version = gallatin::utils::move_to_host(device_version);


			auto vector_log = vector_type::export_to_host(host_version->storage_vector);


			std::vector<std::string> output_strings;

			//printf("Vector log has %lu items\n", vector_log.size());


			for (uint64_t i = 0; i < vector_log.size(); i++){

				
				auto log = vector_log[i];

				//std::cout << log.tid << " " << log.message.length << std::endl;

				// if (log.message.length < 10){
				// 	std::cout << "log " << i << " is busted\n";
				// }

				output_strings.push_back(vector_log[i].export_log());


			}

			//printf("Ouput strings has %lu items", output_strings.size());

			return output_strings;

		}

		//generate output vector and write to host file.
		//for the moment this uses host buffer.
		__host__ void dump_to_file(std::string filename){

			auto log_strings = my_type::export_log(this);

			std::cout << "Writing " << log_strings.size() << " logs to file " << filename << std::endl;


			// for (uint64_t i = 0; i < log_strings.size(); i++){

			// 	if (log_strings[i].size() < 10){
			// 		printf("Small string %lu, length is %lu\n", i, log_strings[i].size());
			// 		std::cout << log_strings[i] << std::endl;
			// 	}

			// 	if (log_strings[i].find("\n") != std::string::npos) {
    		// 		std::cout << "found in " << i << '\n';
			// 	}

			// }

			std::ofstream output_file(filename);
    		std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    		std::copy(log_strings.begin(), log_strings.end(), output_iterator);

		}




	};


}


}


#endif //end of queue name guard