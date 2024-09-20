#ifndef BLOCK_HELPER
#define BLOCK_HELPER


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

struct block_wrapper{

	struct block_int;

	block_int * pimpl;
	uint64_t * bits;

	block_wrapper();
	~block_wrapper();

	void reset_block(uint16_t tree_size);


	bool testReset();
	bool testSingleThread();
	bool testMultiThread();
	bool testMultiRounds();
	bool testInvalidOne();
	bool testInvalidTwo();

	uint readMallocCount();

	uint readFreeCount();


	void open_global_log();

	int close_global_log();

	uint64_t malloc(uint increment_amount);
	bool free();




};


#endif