#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

// INTEGRATION TESTS
class MonteCarloFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const auto &full = info.param;
    const TestType &t = std::get<2>(full);
    std::string size_name = std::get<1>(t);
    std::string task_name = std::get<1>(full);
    return task_name + "_" + size_name;
  }

 protected:
  void SetUp() override {
    auto full_param = GetParam();
    TestType t = std::get<2>(full_param);

    int samples = std::get<0>(t);

    input_.a = {0.0};
    input_.b = {1.0};
    input_.samples = samples;
    input_.func = [](const std::vector<double> &x) { return x[0] * x[0]; };
  }

  bool CheckTestOutputData(OutType &out) final {
    return std::abs(out - (1.0 / 3.0)) < 0.05;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

TEST_P(MonteCarloFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kParams = {
    std::make_tuple(1000, "small"),
    std::make_tuple(5000, "medium"),
    std::make_tuple(20000, "large"),
};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationSEQ, InType>(kParams, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationMPI, InType>(kParams, PPC_SETTINGS_example_processes));

INSTANTIATE_TEST_SUITE_P(IntegrationTests, MonteCarloFuncTests, ppc::util::ExpandToValues(kTaskList),
                         MonteCarloFuncTests::PrintTestParam);

// VALIDATION TESTS (SEQ)
/* class MonteCarloSeqValidationTests : public ::testing::Test {};

TEST_F(MonteCarloSeqValidationTests, InvalidNoFunction) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sync all ranks before doing anything
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 100;
    in.func = nullptr;

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_FALSE(op.Validation());
  }

  // ensure all ranks leave the test at the same time
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MonteCarloSeqValidationTests, InvalidBoundsEmpty) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sync all ranks before doing anything
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {};
    in.b = {};
    in.samples = 100;
    in.func = [](const std::vector<double> &) { return 0.0; };

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_FALSE(op.Validation());
  }

  // ensure all ranks leave the test at the same time
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MonteCarloSeqValidationTests, InvalidDifferentSizes) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sync all ranks before doing anything
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {0.0};
    in.b = {0.0, 1.0};
    in.samples = 100;
    in.func = [](const std::vector<double> &) { return 0.0; };

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_FALSE(op.Validation());
  }

  // ensure all ranks leave the test at the same time
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MonteCarloSeqValidationTests, InvalidWrongBounds) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sync all ranks before doing anything
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {1.0};
    in.b = {0.0};
    in.samples = 100;
    in.func = [](const std::vector<double> &) { return 0.0; };

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_FALSE(op.Validation());
  }

  // ensure all ranks leave the test at the same time
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MonteCarloSeqValidationTests, InvalidSamplesNonPositive) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sync all ranks before doing anything
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 0;
    in.func = [](const std::vector<double> &) { return 0.0; };

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_FALSE(op.Validation());
  }

  // ensure all ranks leave the test at the same time
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MonteCarloSeqValidationTests, Valid) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    InType in;
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 10;
    in.func = [](const std::vector<double> &x) { return x[0]; };

    DorofeevIMonteCarloIntegrationSEQ op(in);
    EXPECT_TRUE(op.Validation());
  }

  MPI_Barrier(MPI_COMM_WORLD);
} */

/* // VALIDATION TESTS (MPI)
class MonteCarloMpiValidationTests : public ::testing::Test {
 protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  int rank{};
};

static bool ValidateMPI(const InType &root_input) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType in;
  if (rank == 0) {
    in = root_input;
  }

  DorofeevIMonteCarloIntegrationMPI op(in);
  return op.Validation();  // result broadcast inside
}

TEST_F(MonteCarloMpiValidationTests, InvalidNoFunc) {
  InType in;
  if (rank == 0) {
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 100;
    in.func = nullptr;
  }
  EXPECT_FALSE(ValidateMPI(in));
}

TEST_F(MonteCarloMpiValidationTests, InvalidSamplesNonPositive) {
  InType in;
  if (rank == 0) {
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 0;
    in.func = [](const std::vector<double> &) { return 0.0; };
  }
  EXPECT_FALSE(ValidateMPI(in));
}

TEST_F(MonteCarloMpiValidationTests, Valid) {
  InType in;
  if (rank == 0) {
    in.a = {0.0};
    in.b = {1.0};
    in.samples = 10;
    in.func = [](const std::vector<double> &) { return 0.0; };
  }
  EXPECT_TRUE(ValidateMPI(in));
} */

}  // namespace dorofeev_i_monte_carlo_integration_processes
