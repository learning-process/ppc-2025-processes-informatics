#include <gtest/gtest.h>
#include <cstddef>
#include <algorithm>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"
#include "chaschin_v_max_for_each_row/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int kCount = 100;  // размер матрицы: kCount x kCount
  InType inputData_;

  void SetUp() override {
    // создаём квадратную матрицу kCount x kCount
    inputData_.resize(kCount);
    for (int i = 0; i < kCount; ++i) {
      inputData_[i].resize(kCount);
    }

    for (int i = 0; i < kCount; i++) {
      for (int j = 0; j < kCount; j++) {
        inputData_[i][j] = static_cast<float>((i + 1) * (j + 2));
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != inputData_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != *std::max_element(inputData_[i].begin(), inputData_[i].end())) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return inputData_;
  }
};

TEST_P(ChaschinVRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChaschinVMaxForEachRow, ChaschinVMaxForEachRowSEQ>(
    PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChaschinVRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChaschinVRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chaschin_v_max_for_each_row
