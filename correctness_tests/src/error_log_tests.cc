/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gtest/gtest.h>
#include <gallatin_static_funcs.hpp>

// Demonstrate some basic assertions.
TEST(logTest, testGlobalInit) {



  helper_open_global_log();

  int n_failures = helper_close_global_log();

  // Expect equality.
  EXPECT_EQ(n_failures, 0);
}


TEST(logTest, testGlobalRun) {



  helper_open_global_log();

  //launch 10 failures.
  log_n_failures(10);

  int n_failures = helper_close_global_log();



  // Expect equality.
  EXPECT_EQ(n_failures, 10);
}

TEST(logTest, testGlobalRunBig) {



  helper_open_global_log();

  //launch 10 failures.
  log_n_failures(50000);

  int n_failures = helper_close_global_log();



  // Expect equality.
  EXPECT_EQ(n_failures, 50000);
}