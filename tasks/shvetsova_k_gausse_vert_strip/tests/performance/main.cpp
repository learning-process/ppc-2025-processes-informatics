#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::vector<double> kCount_ = {0.0};
  InType input_data_;
  OutType expect_res_;

  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKGaussVertStripMPI, ShvetsovaKGaussVertStripSEQ>(
        PPC_SETTINGS_shvetsova_k_gausse_vert_strip);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKGaussVertStripRunPerfTestProcesses::CustomPerfTestName;

TEST_P(ShvetsovaKGaussVertStripRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKGaussVertStripRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_gausse_vert_strip
