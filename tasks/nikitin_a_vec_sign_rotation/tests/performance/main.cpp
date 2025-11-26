#include <gtest/gtest.h>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_vec_sign_rotation {

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
    ppc::util::MakeAllPerfTasks<InType, NikitinAVecSignRotationMPI, NikitinAVecSignRotationSEQ>(PPC_SETTINGS_nikitin_a_vec_sign_rotation);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_vec_sign_rotation
