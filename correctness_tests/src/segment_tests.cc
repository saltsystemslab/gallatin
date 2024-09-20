/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <segment_helpers.hpp>


class segmentHelperTest : public testing::Test {

public:
   segment_helper_tests test_wrapper;

};

// Demonstrate some basic assertions.
TEST_F(segmentHelperTest, assertSegmentInit) {

  EXPECT_TRUE(test_wrapper.testInit());

}

//test set/reset
TEST_F(segmentHelperTest, assertSegmentSetReset) {

  EXPECT_TRUE(test_wrapper.testSetReset());

}

TEST_F(segmentHelperTest, blockMalloc){

  EXPECT_TRUE(test_wrapper.testBlockMalloc());

}

TEST_F(segmentHelperTest, blockMallocFail){

  EXPECT_TRUE(test_wrapper.testBlockMallocFail());

}


TEST_F(segmentHelperTest, testClaimQueue){

  EXPECT_TRUE(test_wrapper.testClaimAllLoop());

}


TEST_F(segmentHelperTest, testParallel){

  EXPECT_TRUE(test_wrapper.testParallel());

}