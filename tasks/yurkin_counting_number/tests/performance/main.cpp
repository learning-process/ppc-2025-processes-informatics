#include <gtest/gtest.h>

#include <cctype>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount_ = 5'000'000;
  InType input_data_{};

  void SetUp() override {
    input_data_.assign(kCount_, '5');
    for (int i = 1; i < kCount_; i += 2) {
      input_data_[i] = 'a';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == kCount_ / 2;
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
