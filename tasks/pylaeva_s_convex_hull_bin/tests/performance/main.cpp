#include <gtest/gtest.h>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "pylaeva_s_convex_hull_bin/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_convex_hull_bin/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_convex_hull_bin {

class PylaevaSConvexHullBinPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(PylaevaSConvexHullBinPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PylaevaSConvexHullBinMPI, PylaevaSConvexHullBinSEQ>(PPC_SETTINGS_pylaeva_s_convex_hull_bin);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSConvexHullBinPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSConvexHullBinPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_convex_hull_bin
