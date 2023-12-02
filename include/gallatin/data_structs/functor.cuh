#ifndef GALLATIN_FUNCTOR
#define GALLATIN_FUNCTOR


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>

#include <cuda/std/tuple>

namespace gallatin {

namespace data_structs {


	template <auto & FN, typename ... Args>
	struct functor {

		using tuple_type = cuda::std::tuple<Args...>;

		tuple_type my_args;

		__device__ functor(Args...input_arguments){

			my_args = cuda::std::make_tuple(input_arguments...);

		}

		__device__ auto apply_args(){


			//printf("%d\n", std::cuda::get<0>(t));

			return cuda::std::apply(FN, my_args);

		}

	};


	//
	template <auto & FN>
	struct deferred_functor {

		//using tuple_type = cuda::std::tuple<Args...>;

		//tuple_type my_args;

		__device__ functor(Args...input_arguments){

			my_args = cuda::std::make_tuple(input_arguments...);

		}

		template <typename ... Args>
		static __device__ auto apply_args(cuda::std::tuple<Args...> my_arguments){


			//printf("%d\n", std::cuda::get<0>(t));

			return cuda::std::apply(FN, my_arguments);

		}

	};


	//demo shows that functor wrapping is valid If they share the same function template
	// template <auto & FN>
	// struct temp_functor {

	// 	template <typename ... Args>
	// 	__device__ void apply_args(Args...arguments){


	// 		auto t = cuda::std::make_tuple(arguments...);

	// 		//printf("%d\n", std::cuda::get<0>(t));

	// 		cuda::std::apply(FN, t);

	// 	}

	// };



}


}


#endif //end of queue name guard