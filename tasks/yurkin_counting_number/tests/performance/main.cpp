#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kCount = 100;
  InType input_data_;

  void SetUp() override {
    input_data_.clear();
    input_data_.reserve(kCount * 100000);

    for (int i = 0; i < kCount * 100000; ++i) {
      input_data_.push_back((i % 3 == 0) ? 'A' : '1');
    }

    GlobalData::g_data_string.clear();
    GlobalData::g_data_string = input_data_;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &) override {
    return true;
  }
};

TEST_P(YurkinCountingNumberPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = YurkinCountingNumberPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinCountingNumberPerfTest, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
