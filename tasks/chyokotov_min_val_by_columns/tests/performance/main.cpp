#include <gtest/gtest.h>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"
#include "chyokotov_min_val_by_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chyokotov_min_val_by_columns {

class ChyokotovMinValPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ChyokotovMinValPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChyokotovMinValByColumnsMPI, ChyokotovMinValByColumnsSEQ>(PPC_SETTINGS_chyokotov_min_val_by_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChyokotovMinValPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChyokotovMinValPerfTest, kGtestValues, kPerfTestName);

}  // namespace chyokotov_min_val_by_columns
