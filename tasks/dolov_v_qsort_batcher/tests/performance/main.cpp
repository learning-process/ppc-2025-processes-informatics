#include <gtest/gtest.h>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(DolovVQsortBatcherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DolovVQsortBatcherMPI, DolovVQsortBatcherSEQ>(
    PPC_SETTINGS_dolov_v_qsort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DolovVQsortBatcherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DolovVQsortBatcherPerfTests, kGtestValues, kPerfTestName);

}  // namespace dolov_v_qsort_batcher
