#ifndef GALLATIN_VISUALIZE
#define GALLATIN_VISUALIZE


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>


//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


namespace gallatin {

namespace data_structs {

	struct color {
		char red;
		char green;
		char blue;
	};


	//output stream that generates a colored 2d mesh given an input stream 
	struct visualizer {

		//std::string output_filename;

		//std::iostream output_file;

		std::vector<std::vector<color>> data_array;

		int width;
		int height;

		visualizer(std::vector<std::vector<color>> input_data){

			data_array = input_data;

			height = data_array.size();

			width = data_array[0].size();


		}

		void write_to_file(std::string output_filename){

			std::ofstream myfile;

			myfile.open (output_filename + ".ppm");

  			myfile << "P6\n";
  			myfile << height << " " << width << "\n";
  			myfile << "255\n";


  			for (int i = 0; i < height; i++){

  				int internal_width = data_array[i].size();

  				for (int j = 0; j < internal_width; j++){

  					color mycolor = data_array[i][j];
  					myfile << mycolor.red << mycolor.green << mycolor.blue;

  				}

  				//output junk color if too big
  				for (int j = internal_width; j < width; j++){

  					color mycolor{ (char) 0, (char) 0, (char) 0};
  					myfile << mycolor.red << mycolor.green << mycolor.blue;
  				}

  				//myfile << '\n';

  			}

  			myfile.close();

  			printf("File write finished\n");

			//std::fstream(output_filename + ".ppm");



		}

	};


}


}


#endif //end of queue name guard