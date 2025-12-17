#include <gtest/gtest.h>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return input_data_ == output_data;
  }

  InType GetTestInputData() override {
    return input_data_;
  }
};

TEST_P(PylaevaSSimpleIterationMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PylaevaSSimpleIterationMethodMPI, PylaevaSSimpleIterationMethodSEQ>(
        PPC_SETTINGS_pylaeva_s_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSSimpleIterationMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSSimpleIterationMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_simple_iteration_method
