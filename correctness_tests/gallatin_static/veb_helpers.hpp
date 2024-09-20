#ifndef VEB_HELPER
#define VEB_HELPER


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

struct veb_helper_tests{


	// veb_helper_tests();
	// ~veb_helper_tests();

	
	bool testComponentSizes();
	bool testInitFFS();
	bool testAtomics();
	bool testFfsAtomic();
	bool testExcepts();
	bool testSetUnset();
	bool testLdAcq();
	bool testGroupSet();
	bool testClaimFirst();
	bool testVebInit();
	bool testVebBasicOps();
	bool testVebFindFirst();
	bool testVebClaimFirst();
	bool testVebParallel();


	void open_global_log();

	int close_global_log();




};


#endif