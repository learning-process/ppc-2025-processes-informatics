#include <gtest/gtest.h>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_dot_product_of_vectors {

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
    ppc::util::MakeAllPerfTasks<InType, SpichekDDotProductOfVectorsMPI, SpichekDDotProductOfVectorsSEQ>(PPC_SETTINGS_spichek_d_dot_product_of_vectors);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace spichek_d_dot_product_of_vectors
