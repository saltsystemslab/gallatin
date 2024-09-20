/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <bs_helpers.hpp>


class bsHelperTest : public testing::Test {

public:
   bs_helper_tests test_wrapper;

};

// Demonstrate some basic assertions.
TEST_F(bsHelperTest, testInit) {

  EXPECT_TRUE(test_wrapper.testStorageInit());

}

TEST_F(bsHelperTest, testPacking) {

  EXPECT_TRUE(test_wrapper.testPacking());

}

TEST_F(bsHelperTest, testSetUnset) {

  EXPECT_TRUE(test_wrapper.testSetUnset());

}