/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>


#include <gallatin_block_tests.hpp>

class BlockTest : public testing::Test {

public:
   block_wrapper test_wrapper;

};


TEST_F(BlockTest, testReset){

   EXPECT_TRUE(test_wrapper.testReset());

}


// TEST_F(BlockTest, testSingleThread){

//    EXPECT_TRUE(test_wrapper.testSingleThread());

// }


TEST_F(BlockTest, testInvalidOne){

   EXPECT_TRUE(test_wrapper.testInvalidOne());

}

TEST_F(BlockTest, testInvalidTwo){

   EXPECT_TRUE(test_wrapper.testInvalidTwo());

}



TEST_F(BlockTest, testMultiThread){

   EXPECT_TRUE(test_wrapper.testMultiThread());

}

TEST_F(BlockTest, testMultiRounds){

   EXPECT_TRUE(test_wrapper.testMultiRounds());

}

// // Demonstrate some basic assertions.
// TEST(BlockTest, BasicAssertions) {
//   // Expect two strings not to be equal.
//   EXPECT_STRNE("hello", "world");
//   // Expect equality.
//   EXPECT_EQ(7 * 6, 42);
// }