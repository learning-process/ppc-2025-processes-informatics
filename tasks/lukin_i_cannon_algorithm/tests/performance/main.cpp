#include <gtest/gtest.h>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lukin_i_cannon_algorithm {
const double EPSILON = 1e-9;

class LukinIRunPerfTestsProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType expected_data_{};

  const int matrix_size_ = 1000;

  void SetUp() override {
    std::vector<double> A(matrix_size_ * matrix_size_);
    std::vector<double> B(matrix_size_ * matrix_size_);

    for (int i = 0; i < A.size(); i++) {
      A[i] = i;
      B[i] = ((A.size() - 1) - i);
    }

    input_data_ = std::make_tuple(A, B, matrix_size_);

    std::vector<double> C(matrix_size_ * matrix_size_);

    for (int i = 0; i < matrix_size_; i++) {
      for (int k = 0; k < matrix_size_; k++) {
        double fixed = A[i * matrix_size_ + k];
        for (int j = 0; j < matrix_size_; j++) {
          C[i * matrix_size_ + j] += fixed * B[k * matrix_size_ + j];
        }
      }
    }
    expected_data_ = C;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (int i = 0; i < expected_data_.size(); i++) {
      if (std::abs(expected_data_[i] - output_data[i]) > EPSILON) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LukinIRunPerfTestsProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LukinICannonAlgorithmMPI, LukinICannonAlgorithmSEQ>(
    PPC_SETTINGS_lukin_i_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LukinIRunPerfTestsProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, LukinIRunPerfTestsProcesses2, kGtestValues, kPerfTestName);

}  // namespace lukin_i_cannon_algorithm
