#include <gtest/gtest.h>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_max_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKRunPerfTestsMatrixMaxElem : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FatehovKRunPerfTestsMatrixMaxElem, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, FatehovKMatrixMaxElemMPI, FatehovKMatrixMaxElemSEQ>(PPC_SETTINGS_fatehov_k_matrix_max_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FatehovKRunPerfTestsMatrixMaxElem::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FatehovKRunPerfTestsMatrixMaxElem, kGtestValues, kPerfTestName);

}  // namespace fatehov_k_matrix_max_elem
