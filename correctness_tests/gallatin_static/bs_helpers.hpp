#ifndef BS_HELPER
#define BS_HELPER


/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#include <stdint.h>
#include <gallatin_static_funcs.hpp>

struct bs_helper_tests{


	// bs_helper_tests();
	// ~bs_helper_tests();

	bool testStorageInit();
	bool testPacking();
	bool testSetUnset();

	void open_global_log();

	int close_global_log();




};


#endif