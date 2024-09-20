/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <gallatin_helpers.hpp>


class fullAllocTest : public testing::Test {

public:
   gallatin_tests test_wrapper;

};

// Demonstrate some basic assertions.
// TEST_F(fullAllocTest, testInit) {

//   EXPECT_TRUE(test_wrapper.testAllocInit());

// }



// TEST_F(fullAllocTest, singleThreadAlloc) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocSingletons());

// }

// TEST_F(fullAllocTest, miniAlloc) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocMini());

// }


// TEST_F(fullAllocTest, testAlloc) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocSingle());

// }

// TEST_F(fullAllocTest, testAllocFree) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocFree());

// }


// TEST_F(fullAllocTest, testAllocFreeMultiSize) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocFreeAllSizes());

// }

TEST_F(fullAllocTest, testAllocFullMalloc) {

  EXPECT_TRUE(test_wrapper.testSliceAllocFreeMalloc());

}



// TEST_F(fullAllocTest, testAllocFreeSingleton) {

//   EXPECT_TRUE(test_wrapper.testSliceAllocFreeSingleton());

// }


