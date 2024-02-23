/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/data_structs/visualizer.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::data_structs;



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   printf("Test started\n");

   std::vector<std::vector<color>> colormap;

   for (int i = 0; i < 256; i++){

      //printf("Started on index %d\n", i)

      std::vector<color> internal_colormap;

      for (int j = 0; j < 256; j++){


         int max = (i+j);
         if (max > 255) max = 510-max;

         //int max = (i+j) % 255;


         color mycolor{ (char) max, (char) 0, (char) (255-max)};

         internal_colormap.push_back(mycolor);


      }

      colormap.push_back(internal_colormap);

   }

   printf("Image written\n");

   visualizer my_visualizer(colormap);

   my_visualizer.write_to_file("test_image");

   return 0;

}
