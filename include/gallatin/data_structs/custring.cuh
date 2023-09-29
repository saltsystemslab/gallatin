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


		//move constructor.
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


		//helpers for parameter pack - efficiently deduce type
		__device__ static uint64_t get_maximum_size(custring alt_string){

			return alt_string.get_len();

		}

		//18446744073709551615 is the max uint size
		__device__ static uint64_t get_maximum_size(uint64_t big_int){

			uint64_t length = 0;


		    if (big_int == 0) length++;

			while(big_int != 0) {
		      big_int = big_int / 10;
		      length++;
			}

			return length;

		}

		__device__ static uint64_t get_maximum_size(double number){


			uint64_t length = 0;


			int precision=5;

			//chars = (char *) global_malloc(MAX_FLOAT_TRUNCATE);

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
		        length += 1;
		    } else { // positive numbers
		        decimals = (uint)((number * cutoff)+.5) % cutoff;
		        units = (uint)number;
		    }


		    //clip end of decimals
		    while (decimals > 0 && decimals % 10 == 0 && decimals != 0)
			{
			    decimals = decimals / 10;
			}

			return length + 1 + get_maximum_size(units) + get_maximum_size(decimals);

		}

		__device__ static uint64_t get_maximum_size(float big_double){

			return get_maximum_size((double) big_double);

		}

		//4294967295	
		__device__ static uint64_t get_maximum_size(uint my_int){

			return get_maximum_size((uint64_t) my_int);

		}

		//-4294967295	
		__device__ static uint64_t get_maximum_size(int my_int){

			return get_maximum_size(int64_t (my_int));

		}

		//-18446744073709551615 is the negative of max uint size
		__device__ static uint64_t get_maximum_size(int64_t big_int){


			if (big_int < 0){
				return 1 + get_maximum_size((uint64_t) (big_int*-1));
			} else {
				return get_maximum_size((uint64_t) big_int);
			}
			//return 21ULL;

		}

		__device__ static uint64_t get_maximum_size(const char * ext_string){

			//calculate length
			uint64_t length = 0;

			if (ext_string == nullptr){
				return 0;
			}

			while (ext_string[length] != '\000' && ext_string[length] != '\0'){

				length+=1;

			}

			return length;

		}

		__device__ static uint64_t get_maximum_size(char * ext_string){

			//calculate length
			uint64_t length = 0;

			if (ext_string == nullptr){
				return 0;
			}

			while (ext_string[length] != '\000' && ext_string[length] != '\0'){

				length+=1;

			}

			return length;

		}


		//and helper to construct one large empty string for make_string
		__device__ static custring make_empty_string(uint64_t max_length){

			custring empty_string;

			if (max_length == 0) return;

			//one extra byte for \0 endstr
			char * data = (char *) global_malloc(max_length+1);

			data[0] = '\0';
			empty_string.set_str(data);
			empty_string.set_len(max_length);

			return empty_string;
		}



		// __device__ void add_to_string(char * ext_string){

		// 	//calculate length

		// 	//uint64_t string_length = custring::get_maximum_size(ext_string);

		// 	uint64_t starting_length = length;

		// 	uint64_t index = 0;

		// 	while (ext_string[index] != '\0'){

		// 		chars[starting_length+index] = ext_string[index];
		// 		index++;

		// 	}

		// 	set_len(starting_length+index);

		// 	chars[length] = '\0';

		// 	return;


		// }


		__device__ uint64_t add_to_string (uint64_t starting_length, const char * ext_string){

			//calculate length

			//uint64_t string_length = custring::get_maximum_size(ext_string);


			uint64_t index = 0;

			while (ext_string[index] != '\0'){

				chars[starting_length+index] = ext_string[index];
				index++;

			}

			

			//chars[length] = '\0';

			return starting_length + index;


		}


		__device__ uint64_t add_to_string (uint64_t starting_length, uint64_t number){

			//calculate length

			//uint64_t string_length = custring::get_maximum_size(ext_string);


		    length = 0;

		    uint64_t num_copy = number;

		    if (number == 0) length++;

			while(num_copy != 0) {
		      num_copy = num_copy / 10;
		      length++;
			}


			//chars = (char *) global_malloc(length+1);

			//chars[length] = '\0';

			if (number == 0) chars[starting_length] = '0';

			int index = starting_length+length-1;

			while (number != 0){
				int digit = number % 10;
				number = number/10;

				chars[index] = '0' + digit;

				index--;

			}


			return starting_length+length;

		}


		__device__ uint64_t add_to_string(uint64_t starting_length, double number){


			int precision=5;

			//chars = (char *) global_malloc(MAX_FLOAT_TRUNCATE);


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


		    //clip end of decimals
		    while (decimals > 0 && decimals % 10 == 0 && decimals != 0)
			{
			    decimals = decimals / 10;
			}

			uint64_t new_length = starting_length;

			if (number < 0){

				new_length = add_to_string(new_length, "-");

			}

		    new_length = add_to_string(new_length, (uint64_t) units);
		    new_length = add_to_string(new_length, ".");
		    new_length = add_to_string(new_length, (uint64_t) decimals);

		    return new_length;

		}


		__device__ uint64_t add_to_string(uint64_t starting_length, float number){

			return add_to_string(starting_length, (double) number);

		}

		__device__ uint64_t add_to_string(uint64_t starting_length, uint number){

			return add_to_string(starting_length, (uint64_t) number);

		}

		__device__ uint64_t add_to_string(uint64_t starting_length, int number){

			uint64_t new_len = starting_length;

			uint clipped_number = (uint) number;

			if (number < 0){

				new_len = add_to_string(new_len, "-");
				clipped_number = (uint) (number*-1);


			}

			return add_to_string(new_len, clipped_number);

		}




	};


	template <typename Last>
	__device__ uint64_t custring_est_size (Last last) {

	    return custring::get_maximum_size(last);

	}

	template <typename First, typename Second, typename...Rest>
	__device__ uint64_t custring_est_size(First first_item, Second second_item, Rest...remaining){

		return custring_est_size(first_item) + custring_est_size<Second, Rest...>(second_item, remaining...);

	}

	template <typename Last>
	__device__ uint64_t add_to_string_variadic(custring & target, uint64_t length, Last last){

		return target.add_to_string(length, last);

	}

	template <typename First, typename Second, typename...Rest>
	__device__ uint64_t add_to_string_variadic(custring & target, uint64_t length, First first_item, Second second_item, Rest...remaining){

		uint64_t intermediate_length = add_to_string_variadic(target, length, first_item);
		return add_to_string_variadic<Second, Rest...>(target, intermediate_length, second_item, remaining...);

	}

	//use parameter pack to efficiently add multiple strings together
	//procedure - grok size of all 
	template <typename ... Args>
	__device__ custring make_string(Args...all_args){

		uint64_t size = custring_est_size<Args...>(all_args...);

		//printf("Size is bounded by %lu\n", size);


		custring test_string = custring::make_empty_string(size);

		uint64_t length = add_to_string_variadic(test_string, 0, all_args...);

		test_string.set_len(length);

		test_string.data()[length] = '\0';

		return test_string;


	}


}


}


#endif //end of string name guard