#ifndef SEGMENT_HELPER
#define SEGMENT_HELPER


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

struct segment_helper_tests{
	
	bool testInit();
	bool testSetReset();
	bool testBlockMalloc();
	bool testBlockMallocFail();
	bool testClaimAllLoop();

	bool testParallel();

	void open_global_log();

	int close_global_log();




};


#endif