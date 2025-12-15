#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "egashin_k_iterative_simple/mpi/include/ops_mpi.hpp"
#include "egashin_k_iterative_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egashin_k_iterative_simple {

class EgashinKIterativeSimplePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kN_ = 5000;
  InType input_;
  OutType expected_;

  void SetUp() override {
    std::vector<std::vector<double>> A(kN_, std::vector<double>(kN_, 0.0));
    std::vector<double> x0(kN_, 0.0);

    for (int i = 0; i < kN_; ++i) {
      A[i][i] = 4.0;
      if (i > 0) {
        A[i][i - 1] = -1.0;
      }
      if (i < kN_ - 1) {
        A[i][i + 1] = -1.0;
      }
    }

    std::vector<double> b(kN_, 2.0);
    if (kN_ > 0) {
      b[0] = 3.0;
      b[static_cast<std::size_t>(kN_ - 1)] = 3.0;
    }

    input_.A = A;
    input_.b = b;
    input_.x0 = x0;
    input_.tolerance = 1e-6;
    input_.max_iterations = 10000;

    expected_.resize(kN_, 1.0);
  }

  bool CheckTestOutputData(OutType &output) override {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    if (output.size() != expected_.size()) {
      return false;
    }

    double tolerance = input_.tolerance * 100;
    for (std::size_t i = 0; i < output.size(); ++i) {
      if (std::abs(output[i] - expected_[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() override {
    return input_;
  }
};

TEST_P(EgashinKIterativeSimplePerfTest, Performance) {
  ExecuteTest(GetParam());
}

const auto kPerfParams =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_egashin_k_iterative_simple);

const auto kGtestValues = ppc::util::TupleToGTestValues(kPerfParams);

const auto kPerfTestName = EgashinKIterativeSimplePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(EgashinKIterativeSimplePerf, EgashinKIterativeSimplePerfTest, kGtestValues, kPerfTestName);

}  // namespace egashin_k_iterative_simple
