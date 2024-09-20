#ifndef GAL_HELPER
#define GAL_HELPER


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

struct gallatin_tests{


	// bs_helper_tests();
	// ~bs_helper_tests();

	bool testAllocInit();
	bool testSliceAllocSingle();
	bool testSliceAllocMini();
	bool testSliceAllocSingletons();
	bool testSliceAllocFree();
	bool testSliceAllocFreeSingleton();
	bool testSliceAllocFreeAllSizes();
	bool testSliceAllocFreeMalloc();


	void open_global_log();

	int close_global_log();




};


#endif