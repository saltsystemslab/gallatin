#ifndef GALLATIN_CUSTRING
#define GALLATIN_CUSTRING


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/global_allocator.cuh>


#define MAX_FLOAT_TRUNCATE 10

//To build log files, we need strings!
//this type does 

//entries maintain TID, logID, and a message generated via 
namespace gallatin {

namespace data_structs {

	//not necessary
	//const __device__ char digits[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	using namespace gallatin::allocators;

	//merge two strings together, putting them in-place in the first string.
	template <typename custring>
	__device__ custring combine(custring first, custring second){

		uint combined_length = first.get_len() + second.get_len();

		//auto allocator = first.get_alloc();

		char * new_string = (char *) global_malloc(combined_length+1);

		char * first_str = first.data();
		char * second_str = second.data();

		for (int i = 0; i < first.get_len(); i++){
			new_string[i] = first_str[i];
		}

		for (int i=0; i < second.get_len(); i++){

			new_string[i+first.get_len()] = second_str[i]; 

		}

		new_string[combined_length] = '\0';

		//free old strings
		global_free(first_str);
		global_free(second_str);

		first.set_len(combined_length);
		first.set_str(new_string);

		second.set_len(0);
		second.set_str(nullptr);

		__threadfence();

		return first;



	}


	struct custring {


		char * chars;
		uint length;
		

		__device__ custring(){

			length = 0;
			chars = nullptr;

		}

		__device__ custring (char * ext_string){

			//calculate length
			length = 0;

			if (ext_string == nullptr){
				chars = nullptr;
				return;
			}

			while (ext_string[length] != '\000' && ext_string[length] != '\0'){

				length+=1;

			}

			//length+=1;


			chars = (char *) global_malloc(length+1);

			for (int i = 0; i < length; i++){

				chars[i] = ext_string[i];

			}

			chars[length] = '\0';

			return;


		}


		__device__ custring (const char * ext_string){

			//calculate length
			length = 0;

			if (ext_string == nullptr){
				chars = nullptr;
				return;
			}

			while (ext_string[length] != '\000' && ext_string[length] != '\0'){

				length+=1;

			}

			//length+=1;


			chars = (char *) global_malloc(length+1);

			for (int i = 0; i < length; i++){

				chars[i] = ext_string[i];

			}

			chars[length] = '\0';

			return;


		}


		__device__ custring(uint64_t number){


		    length = 0;

		    uint64_t num_copy = number;

		    if (number == 0) length++;

			while(num_copy != 0) {
		      num_copy = num_copy / 10;
		      length++;
			}


			chars = (char *) global_malloc(length+1);

			chars[length] = '\0';

			if (number == 0) chars[0] = '0';

			int index = length-1;

			while (number != 0){
				int digit = number % 10;
				number = number/10;

				chars[index] = '0' + digit;

				index--;

			}


			return;



		}


		__device__ custring(uint number): custring((uint64_t) number){}

		__device__ custring(custring&& other){

			set_len(other.get_len());
			set_str(other.data());

			other.set_str(nullptr);
			other.set_len(0);

		}

		//copy constructor
		__device__ custring(custring & other){

			auto copy_len = other.get_len();

			length = copy_len;

			char * other_data = other.data();

			chars = (char *) global_malloc(copy_len);

			for (int i = 0; i < copy_len; i++){
				chars[i] = other_data[i];
			}

			chars[length] = '\0';


		}

				//copy constructor
		__device__ custring(const custring & other){

			auto copy_len = other.length;

			length = copy_len;

			char * other_data = other.chars;

			chars = (char *) global_malloc(copy_len);

			for (int i = 0; i < copy_len; i++){
				chars[i] = other_data[i];
			}

			chars[length] = '\0';


		}


		__device__ custring(float number, int precision=5){

			chars = (char *) global_malloc(MAX_FLOAT_TRUNCATE);


			int cutoff = 1;

			while (precision > 0){

				cutoff *=10;
				precision--;

			}

		    uint decimals;  // variable to store the decimals
		    uint units;  // variable to store the units (part to left of decimal place)


		    if (number < 0) { // take care of negative numbers
		        decimals = (uint)((number * -cutoff)+.5) % cutoff; // make 1000 for 3 decimals etc.
		        units = (uint)(-1 * number);
		    } else { // positive numbers
		        decimals = (uint)((number * cutoff)+.5) % cutoff;
		        units = (uint)number;
		    }



		    custring above((uint64_t) units);
		    custring below((uint64_t) decimals);
		    custring decimal_str(".");

		    int clipped = 0;
		    for (int i = below.get_len(); i >= 1; i--){
		    	if (below.chars[i] == '0'){
		    		clipped += 1;
		    	}
		    }

		    below.set_len(below.get_len()-clipped);
		    below.chars[below.get_len()] = '\0';

		    custring mixed = above + decimal_str + below;

		    custring(mixed.chars);

			

		}

		__device__ custring(double number, int precision=5){

			chars = (char *) global_malloc(MAX_FLOAT_TRUNCATE);


			int cutoff = 1;

			while (precision > 0){

				cutoff *=10;
				precision--;

			}

		    uint decimals;  // variable to store the decimals
		    uint units;  // variable to store the units (part to left of decimal place)


		    if (number < 0) { // take care of negative numbers
		        decimals = (uint)((number * -cutoff)+.5) % cutoff;  // make 1000 for 3 decimals etc.
		        units = (uint)(-1 * number);
		    } else { // positive numbers
		        decimals = (uint)((number * cutoff)+.5) % cutoff;
		        units = (uint)number;
		    }



		    custring above((uint64_t) units);
		    custring below((uint64_t) decimals);

		    //clip decimals

		    int clipped = 0;
		    for (int i = below.get_len(); i >= 1; i--){
		    	if (below.chars[i] == '0'){
		    		clipped += 1;
		    	}
		    }

		    below.set_len(below.get_len()-clipped);
		    below.chars[below.get_len()] = '\0';

		    custring decimal_str(".");

		    custring mixed = above + decimal_str + below;

		    set_str(mixed.data());
		    set_len(mixed.get_len());



			

		}

		__device__ void print_info(){

			if (length == 0){
				printf("String has no length.\n");
			} else {
				printf("String has length %u, first char %c\n", length, chars[0]);
			}
			

		}

		__device__ void print_string_device(){

			for (int i = 0; i < length; i++){
				printf("%c", chars[i]);
			}

			printf("\n");

		}


		__device__ uint get_len(){
			return length;
		}

		__device__ void set_len(uint new_length){
			length = new_length;
		}

		__device__ char * data(){
			return chars;
		}

		__device__ void set_str(char * ext_chars){
			chars = ext_chars;
		}

		__device__ custring operator+ (const custring & first){

			return combine<custring>(*this, first);

		}

		__device__ custring operator+(const char * ext_str){

			return combine<custring>(*this, custring(ext_str));

		}

		__device__ custring operator=(const custring & first){

			length = first.length;

			chars = first.chars;

		}


		__device__ ~custring(){

			if (chars != nullptr){
				global_free(chars);
			}

			
			
		}



	};


	//use parameter pack to efficiently add multiple strings together
	//procedure - grok size of all 
	template <class ... Args>
	__device__ custring combine_all(custring){



	}


}


}


#endif //end of string name guard