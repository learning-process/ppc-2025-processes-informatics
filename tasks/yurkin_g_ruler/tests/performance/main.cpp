#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "yurkin_g_ruler/common/include/common.hpp"
#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"
#include "yurkin_g_ruler/seq/include/ops_seq.hpp"

namespace yurkin_g_ruler {

class YurkinGRulerPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount = 100000000;
  InType inputData{};

  void SetUp() override {
    inputData = kCount;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return inputData == output_data;
  }

  InType GetTestInputData() final {
    return inputData;
  }
};

TEST_P(YurkinGRulerPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, YurkinGRulerMPI, YurkinGRulerSEQ>(PPC_SETTINGS_yurkin_g_ruler);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YurkinGRulerPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinGRulerPerfTests, kGtestValues, kPerfTestName);

}  // namespace yurkin_g_ruler
