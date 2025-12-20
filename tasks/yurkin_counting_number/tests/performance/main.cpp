#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_;

  void SetUp() override {
    GlobalData::g_data_string.clear();
    GlobalData::g_data_string.resize(5'000'000, 'a');

    for (int i = 0; i < kCount_; i++) {
      GlobalData::g_data_string[i] = 'X';
    }

    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YurkinCountingNumberPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YurkinCountingNumberPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinCountingNumberPerfTests, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
