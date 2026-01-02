#include <gtest/gtest.h>

#include "gusev_d_radix_double/common/include/common.hpp"
#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gusev_d_radix_double {

class GusevDRadixDoublePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(GusevDRadixDoublePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GusevDRadixDoubleMPI, GusevDRadixDoubleSEQ>(PPC_SETTINGS_gusev_d_radix_double);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevDRadixDoublePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevDRadixDoublePerfTests, kGtestValues, kPerfTestName);

}  // namespace gusev_d_radix_double
