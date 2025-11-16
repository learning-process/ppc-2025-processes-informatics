#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"
#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"
#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozova_s_matrix_max_value {

class MorozovaSRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    int size = 500;
    input_data_.resize(size, std::vector<int>(size));
    int value = 1;
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        input_data_[i][j] = value++;
      }
    }
    input_data_[size/2][size/2] = 1000000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 1000000;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MorozovaSRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MorozovaSMatrixMaxValueMPI, MorozovaSMatrixMaxValueSEQ>(PPC_SETTINGS_morozova_s_matrix_max_value);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovaSRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovaSRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozova_s_matrix_max_value