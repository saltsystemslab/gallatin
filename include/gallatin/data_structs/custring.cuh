#ifndef GALLATIN_CUSTRING
#define GALLATIN_CUSTRING


#include <cuda.h>
#include <cuda_runtime_api.h>


#define MAX_FLOAT_TRUNCATE 10

//To build log files, we need strings!
//this type does 

//entries maintain TID, logID, and a message generated via 
namespace gallatin {

namespace data_structs {

	//not necessary
	//const __device__ char digits[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };


	//merge two strings together, putting them in-place in the first string.
	template <typename custring>
	custring combine(custring first, custring second){

		uint combined_length = first.get_len() + second.get_len();

		auto allocator = first.get_alloc();

		char * new_string = allocator->malloc(combined_length);

		char * first_str = first.data();
		char * second_str = second.data();

		for (int i = 0; i < first.get_len(); i++){
			new_string[i] = first_str[i];
		}

		for (int i=0; i < second.get_len(); i++){

			new_string[i+first.get_len()] = second_str[i]; 

		}

		allocator->free(first_str);

		//second calls it's own allocator just in case they differ.
		second.get_alloc()->free(second_str);

		first.set_len(combined_length);
		first.set_data(new_string);

		second.set_len(0);
		second.set_data(nullptr);

		__threadfence();

		return first;



	}


	template <typename Allocator>
	struct custring {


		using full_type = custring<Allocator>;

		char * chars;
		uint length;
		Allocator * alloc;
		

		__device__ custring (char * ext_string, Allocator * ext_alloc){


			alloc = ext_alloc;
			//calculate length
			length = 0;

			if (ext_string == nullptr){
				chars = nullptr;
				return;
			}

			while (ext_string[length] != '\000' && ext_string[length] != '\0'){

				length+=1;

			}

			length+=1;


			chars = (char *) alloc->malloc(length);

			for (int i = 0; i < length; i++){

				chars[i] = ext_string[i];

			}

			return;


		}


		__device__ custring(uint64_t number, Allocator * ext_alloc){


			alloc = ext_alloc;

		    length = 0;

		    uint64_t num_copy = number;

			while(num_copy != 0) {
		      num_copy = num_copy / 10;
		      length++;
			}


			chars = (char *) alloc->malloc(length);


			int index = length-1;

			while (number != 0){
				int digit = number % 10;
				number = number/10;

				chars[index] = '0' + digit;

				index--;

			}


			return;



		}


		__device__ custring(uint number, Allocator * ext_alloc){

			custring((uint64_t) number, ext_alloc);
		}

		__device__ custring(custring&&) = default;


		__device__ custring(float number, Allocator * ext_alloc, int precision=5){

			allocator = ext_alloc;

			chars = (char *) allocator->malloc(MAX_FLOAT_TRUNCATE);


			int cutoff = 1;

			while (precision > 0){

				cutoff *=10;
				precision--;

			}

		    uint decimals;  // variable to store the decimals
		    uint units;  // variable to store the units (part to left of decimal place)


		    if (x < 0) { // take care of negative numbers
		        decimals = (uint)(x * -cutoff) % cutoff; // make 1000 for 3 decimals etc.
		        units = (uint)(-1 * x);
		    } else { // positive numbers
		        decimals = (uint)(x * cutoff) % cutoff;
		        units = (uint)x;
		    }



		    full_type above((uint64_t) units, ext_alloc);
		    full_type below((uint64_t) decimals, ext_alloc);
		    full_type decimal_str(".", ext_alloc);

		    full_type mixed = above + decimal_str + below;

		    custring(mixed);

			

		}

		__device__ void print_info(){

			if (length == 0){
				printf("String has no length.\n");
			} else {
				printf("String has length %u, first char %c\n", length, chars[0]);
			}
			

		}


		__device__ uint get_len(){
			return length;
		}

		__device__ uint set_len(uint new_length){
			length = new_length;
		}

		__device__ char * data(){
			return chars;
		}

		__device__ char * set_str(char * ext_chars){
			chars = ext_chars;
		}

		__device__ Allocator * get_alloc(){
			return alloc;
		}


		__device__ full_type operator+ (const full_type & first){

			return combine<full_type>(*this, first);

		}

		__device__ ~custring(){

			if (chars != nullptr){
				alloc->free(chars);
			}

			
			
		}



	};


}


}


#endif //end of string name guard