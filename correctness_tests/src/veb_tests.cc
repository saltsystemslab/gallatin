/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <veb_helpers.hpp>


class vebHelperTest : public testing::Test {

public:
   veb_helper_tests test_wrapper;

};

// Demonstrate some basic assertions.
TEST_F(vebHelperTest, assertsize) {

  EXPECT_TRUE(test_wrapper.testComponentSizes());

}

TEST_F(vebHelperTest, initFFS) {

  EXPECT_TRUE(test_wrapper.testInitFFS());

}

TEST_F(vebHelperTest, atomicWrites) {

  EXPECT_TRUE(test_wrapper.testAtomics());

}

TEST_F(vebHelperTest, fssAtomicMixed) {

  EXPECT_TRUE(test_wrapper.testFfsAtomic());

}

TEST_F(vebHelperTest, rangeExceptions) {

  EXPECT_TRUE(test_wrapper.testExcepts());

}

TEST_F(vebHelperTest, setUnset) {

  EXPECT_TRUE(test_wrapper.testSetUnset());

}

TEST_F(vebHelperTest, load_acquire) {

  EXPECT_TRUE(test_wrapper.testLdAcq());

}

TEST_F(vebHelperTest, group_set_unset) {

  EXPECT_TRUE(test_wrapper.testGroupSet());

}

TEST_F(vebHelperTest, claim_first) {

  EXPECT_TRUE(test_wrapper.testClaimFirst());

}

TEST_F(vebHelperTest, veb_init) {

  EXPECT_TRUE(test_wrapper.testVebInit());

}

TEST_F(vebHelperTest, veb_basic) {

  EXPECT_TRUE(test_wrapper.testVebBasicOps());

}

TEST_F(vebHelperTest, veb_find_first) {

  EXPECT_TRUE(test_wrapper.testVebFindFirst());

}

TEST_F(vebHelperTest, veb_claim_first) {

  EXPECT_TRUE(test_wrapper.testVebClaimFirst());

}

TEST_F(vebHelperTest, veb_parallel_ops) {

  EXPECT_TRUE(test_wrapper.testVebParallel());

}

