#include <gtest/gtest.h>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_violation_order_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace goriacheva_k_violation_order_elem_vec {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GoriachevaKViolationOrderElemVecMPI, GoriachevaKViolationOrderElemVecSEQ>(PPC_SETTINGS_goriacheva_k_violation_order_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace goriacheva_k_violation_order_elem_vec
