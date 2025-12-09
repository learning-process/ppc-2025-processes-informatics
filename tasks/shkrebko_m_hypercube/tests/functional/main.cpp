#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"

namespace shkrebko_m_hypercube {

class ShkrebkoMHypercubeFuncTest : public ::testing::Test {
 protected:
  static int GetWorldSize() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  static int GetWorldRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  static bool IsPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
  }

  static void CheckHypercubePath(const std::vector<int> &path, int source, int dest) {
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front(), source);
    EXPECT_EQ(path.back(), dest);

    for (size_t i = 1; i < path.size(); i++) {
      int prev = path[i - 1];
      int curr = path[i];
      int diff = prev ^ curr;

      EXPECT_EQ(diff & (diff - 1), 0) << "Invalid hypercube transition from " << prev << " to " << curr
                                      << " (diff: " << diff << ")";
    }
  }

  static bool ShouldSkipTest(int min_processes = 2) {
    int size = GetWorldSize();
    
    bool skip_local = (size < min_processes || !IsPowerOfTwo(size));
    int skip_int = skip_local ? 1 : 0;
    int skip_all = 0;
    MPI_Allreduce(&skip_int, &skip_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    return (skip_all == 1);
  }
};

TEST_F(ShkrebkoMHypercubeFuncTest, RouteFromZeroToOne) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int rank = GetWorldRank();
  InType input = {100, 1};

  ShkrebkoMHypercubeMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 100);
  EXPECT_EQ(output.destination, 1);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    CheckHypercubePath(output.path, 0, 1);
  }
}

TEST_F(ShkrebkoMHypercubeFuncTest, RouteFromZeroToLast) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int size = GetWorldSize();
  int rank = GetWorldRank();
  InType input = {200, size - 1};

  ShkrebkoMHypercubeMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 200);
  EXPECT_EQ(output.destination, size - 1);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    CheckHypercubePath(output.path, 0, size - 1);
  }
}

TEST_F(ShkrebkoMHypercubeFuncTest, RouteToSelf) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int rank = GetWorldRank();
  InType input = {300, 0};

  ShkrebkoMHypercubeMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 300);
  EXPECT_EQ(output.destination, 0);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    EXPECT_EQ(output.path.size(), 1);
    EXPECT_EQ(output.path[0], 0);
  }
}

TEST_F(ShkrebkoMHypercubeFuncTest, SeqRouteFromZeroToOne) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int rank = GetWorldRank();
  InType input = {400, 1};

  ShkrebkoMHypercubeSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 400);
  EXPECT_EQ(output.destination, 1);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    CheckHypercubePath(output.path, 0, 1);
  }
}

TEST_F(ShkrebkoMHypercubeFuncTest, SeqRouteFromZeroToLast) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int size = GetWorldSize();
  int rank = GetWorldRank();
  InType input = {500, size - 1};

  ShkrebkoMHypercubeSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 500);
  EXPECT_EQ(output.destination, size - 1);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    CheckHypercubePath(output.path, 0, size - 1);
  }
}

TEST_F(ShkrebkoMHypercubeFuncTest, SeqRouteToSelf) {
  if (ShouldSkipTest(2)) {
    GTEST_SKIP() << "Hypercube requires at least 2 processes and power of 2";
  }
  
  int rank = GetWorldRank();
  InType input = {600, 0};

  ShkrebkoMHypercubeSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  EXPECT_EQ(output.value, 600);
  EXPECT_EQ(output.destination, 0);
  EXPECT_TRUE(output.finish);

  if (rank == 0) {
    EXPECT_EQ(output.path.size(), 1);
    EXPECT_EQ(output.path[0], 0);
  }
}



}  // namespace shkrebko_m_hypercube