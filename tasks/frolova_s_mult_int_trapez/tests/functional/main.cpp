#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <random>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"
#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

std::pair<double, double> GetRandomLimit(double min_value, double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::pair<double, double> result;
  max_value = max_value * 1000 - 5;
  min_value *= 1000;
  result.first = (double)(gen() % (int)(max_value - min_value) + min_value) / 1000;
  max_value += 5;
  min_value = result.first * 1000;
  result.second = (double)(gen() % (int)(max_value - min_value) + min_value) / 1000;
  return result;
}

unsigned int GetRandomIntegerData(unsigned int min_value, unsigned int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return gen() % (max_value - min_value) + min_value;
}

double function1(std::vector<double> input) {
  return pow(input[0], 3) + pow(input[1], 3);
}
double function2(std::vector<double> input) {
  return sin(input[0]) + sin(input[1]) + sin(input[2]);
}
double function3(std::vector<double> input) {
  return 8 * input[0] * input[1] * input[2];
}
double function4(std::vector<double> input) {
  return -1.0 / sqrt(1 - pow(input[0], 2));
}
double function5(std::vector<double> input) {
  return -(sin(input[0]) * cos(input[1]));
}
double function6(std::vector<double> input) {
  return (-3 * pow(input[1], 2) * sin(5 * input[0])) / 2;
}

// SEQ Tests
TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_1) {
  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}, {1.0, 3.2}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function1};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_GT(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_2) {
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {80, 80, 80};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function2};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_NE(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_3) {
  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {80, 80, 80};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function3};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_GT(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_4) {
  std::vector<std::pair<double, double>> limits = {{0.0, 0.5}};
  std::vector<unsigned int> intervals = {1000};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function4};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_LT(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_5) {
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function5};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_LT(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_6) {
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function6};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_NE(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_random_limits) {
  std::vector<std::pair<double, double>> limits = {GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0),
                                                   GetRandomLimit(0.0, 10.0)};
  std::vector<unsigned int> intervals = {80, 80, 80};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function3};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_GT(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_random_intervals) {
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {GetRandomIntegerData(100, 150), GetRandomIntegerData(100, 150)};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function6};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_NE(result, 0.0);
}

TEST(frolova_s_star_topology_seq_functional, Test_of_functionality_random_limits_and_intervals) {
  std::vector<std::pair<double, double>> limits = {GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0),
                                                   GetRandomLimit(0.0, 10.0)};
  std::vector<unsigned int> intervals = {GetRandomIntegerData(40, 60), GetRandomIntegerData(40, 60),
                                         GetRandomIntegerData(40, 60)};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function3};
  frolova_s_star_topology::FrolovaSStarTopologySEQ task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  double result = task.GetOutput();
  EXPECT_GT(result, 0.0);
}

// MPI Tests
TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_1) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}, {1.0, 3.2}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function1};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_GT(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_2) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {80, 80, 80};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function2};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_NE(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_3) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {80, 80, 80};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function3};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_GT(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_4) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 0.5}};
  std::vector<unsigned int> intervals = {1000};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function4};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_LT(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_5) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function5};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_LT(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_6) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {100, 100};

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function6};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_NE(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_random_limits) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals = {80, 80, 80};

  if (rank == 0) {
    limits = {GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0)};
  }

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function3};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_GT(result, 0.0);
  }
}

TEST(frolova_s_star_topology_mpi_functional, Test_of_functionality_random_intervals) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals;

  if (rank == 0) {
    intervals = {GetRandomIntegerData(100, 150), GetRandomIntegerData(100, 150)};
  }

  frolova_s_star_topology::TrapezoidalIntegrationInput input{limits, intervals, function6};
  frolova_s_star_topology::FrolovaSStarTopologyMPI task(input);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (rank == 0) {
    double result = task.GetOutput();
    EXPECT_NE(result, 0.0);
  }
}
