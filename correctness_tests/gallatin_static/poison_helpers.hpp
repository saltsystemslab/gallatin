#ifndef POISON_HELPER
#define POISON_HELPER


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

struct poison_helper_tests{


	// veb_helper_tests();
	// ~veb_helper_tests();

	bool testInit();
	bool testError();
	bool testErrorMulti();
	bool testArray();


	void open_global_log();

	int close_global_log();




};


#endif