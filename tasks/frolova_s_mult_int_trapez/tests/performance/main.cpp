#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"
#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

using namespace frolova_s_mult_int_trapez;

double function1(std::vector<double> input) {
  return pow(input[0], 3) + pow(input[1], 3);
}
double function2(std::vector<double> input) {
  return (-3 * pow(input[1], 2) * sin(5 * input[0])) / 2;
}

// SEQ Performance Tests
TEST(frolova_s_mult_int_trapez_seq_perf, test_small_problem) {
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {100, 100};

  TrapezoidalIntegrationInput input{limits, intervals, function2};
  FrolovaSMultIntTrapezSEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  double result = task.GetOutput();
  EXPECT_NE(result, 0.0);

  std::chrono::duration<double> duration = end - start;
  std::cout << "SEQ Small problem time: " << duration.count() << " seconds" << std::endl;
}

TEST(frolova_s_mult_int_trapez_seq_perf, test_large_problem) {
  std::vector<std::pair<double, double>> limits = {{0.0, 2.0}, {0.0, 2.0}};
  std::vector<unsigned int> intervals = {500, 500};

  TrapezoidalIntegrationInput input{limits, intervals, function1};
  FrolovaSMultIntTrapezSEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  double result = task.GetOutput();
  EXPECT_NE(result, 0.0);

  std::chrono::duration<double> duration = end - start;
  std::cout << "SEQ Medium problem time: " << duration.count() << " seconds" << std::endl;
}

TEST(frolova_s_star_topology_seq_perf, test_large_problem) {
  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {100, 100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function1};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  double result = task.GetOutput();
  EXPECT_GT(result, 0.0);

  std::chrono::duration<double> duration = end - start;
  std::cout << "SEQ Large problem time: " << duration.count() << " seconds" << std::endl;
}

// MPI Performance Tests
TEST(frolova_s_mult_int_trapez_mpi_perf, test_large_problem) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 2.0}, {0.0, 2.0}};
  std::vector<unsigned int> intervals = {500, 500};

  TrapezoidalIntegrationInput input{limits, intervals, function1};
  FrolovaSMultIntTrapezMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_NE(result, 0.0);
    std::chrono::duration<double> duration = end - start;
    std::cout << "MPI Small problem time: " << duration.count() << " seconds" << std::endl;
  }
}

TEST(frolova_s_mult_int_trapez_mpi_perf, test_medium_problem) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {300, 300};

  TrapezoidalIntegrationInput input{limits, intervals, function2};
  FrolovaSMultIntTrapezMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_NE(result, 0.0);
    std::chrono::duration<double> duration = end - start;
    std::cout << "MPI Medium problem time: " << duration.count() << " seconds" << std::endl;
  }
}

TEST(frolova_s_star_topology_mpi_perf, test_large_problem) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {100, 100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function1};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
  auto end = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_GT(result, 0.0);
    std::chrono::duration<double> duration = end - start;
    std::cout << "MPI Large problem time: " << duration.count() << " seconds" << std::endl;
  }
}
