#include <gtest/gtest.h>

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"
#include "nikitin_a_buble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_buble_sort {

class NikitinAPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(NikitinAPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinABubleSortMPI, NikitinABubleSortSEQ>(PPC_SETTINGS_nikitin_a_buble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinAPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinAPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_buble_sort
