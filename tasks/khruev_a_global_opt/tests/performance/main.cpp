#include <gtest/gtest.h>

#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"
#include "khruev_a_global_opt/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data_;

  void SetUp() override {
    input_data_.func_id = 2;
    input_data_.ax = 0.0;
    input_data_.bx = 1.0;
    input_data_.ay = 0.0;
    input_data_.by = 1.0;

    input_data_.epsilon = 1e-7;
    input_data_.max_iter = 1000;
    input_data_.r = 2.2;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.value < 1e10;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(KhruevAGlobalOptPerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KhruevAGlobalOptMPI, KhruevAGlobalOptSEQ>(PPC_SETTINGS_khruev_a_global_opt);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KhruevAGlobalOptPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KhruevAGlobalOptPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace khruev_a_global_opt
