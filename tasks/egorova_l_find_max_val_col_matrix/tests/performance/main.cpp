#include <gtest/gtest.h>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"
#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(EgorovaLRunPerfTestProcesses, EgorovaLRunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EgorovaLFindMaxValColMatrixMPI, EgorovaLFindMaxValColMatrixSEQ>(
        PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EgorovaLRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(EgorovaLRunModeTests, EgorovaLRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace egorova_l_find_max_val_col_matrix
