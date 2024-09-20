/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <poison_helpers.hpp>


class poisonTest : public testing::Test {

public:
   poison_helper_tests test_wrapper;

};


TEST_F(poisonTest, testPoisonInit) {

  EXPECT_TRUE(test_wrapper.testInit());

}

TEST_F(poisonTest, testPoisonError) {

  EXPECT_TRUE(test_wrapper.testError());

}

TEST_F(poisonTest, testPoisonMultiError) {

  EXPECT_TRUE(test_wrapper.testErrorMulti());

}

TEST_F(poisonTest, testArray) {

  EXPECT_TRUE(test_wrapper.testArray());

}



