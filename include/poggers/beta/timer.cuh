#ifndef BETA_TIMER
#define BETA_TIMER
//Betta, the block-based extending-tree thread allocaotor, made by Hunter McCoy (hunter@cs.utah.edu)
//Copyright (C) 2023 by Hunter McCoy

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//and associated documentation files (the "Software"), to deal in the Software without restriction, 
//including without l> imitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
//LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//The alloc table is an array of uint64_t, uint64_t pairs that store



//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>


using namespace std::chrono;

namespace beta {

namespace utils {

struct timer {

	uint64_t_bitarr lock_bits;

	high_resolution_clock::time_point start;

	high_resolution_clock::time_point end;

	//flush device and start timer
	timer(){

		cudaDeviceSynchronize();
		start_timer();
	}

	__host__ double elapsed() {
   		return (duration_cast<duration<double> >(end - start)).count();
	}

	__host__ void start_timer(){

		start = high_resolution_clock::now();

	}

	__host__ void end_timer(){

		end = high_resolution_clock::now();

	}

	//synchronize with device, end the timer, and report duration
	__host__ double sync_end(){

		cudaDeviceSynchronize();

		end_timer();

		return elapsed();

	}


	__host__ void print_throughput(std::string operation, uint64_t nitems){

		std::cout << operation << " " <<  nitems << " in " << elapsed() << " seconds, throughput " << std::fixed << 1.0*nitems/elapsed() << std::endl;   
      

	}


};

}

}


#endif //End of VEB guard