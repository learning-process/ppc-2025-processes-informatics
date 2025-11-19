#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"
#include "chaschin_v_max_for_each_row/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_Count = 5000;  // размер матрицы: k_Count x k_Count
  InType input_data;

  void SetUp() override {
    // создаём квадратную матрицу k_Count x k_Count
    input_data.resize(k_Count);
    for (int i = 0; i < k_Count; ++i) {
      input_data[i].resize(k_Count);
    }

    for (int i = 0; i < k_Count; i++) {
      for (int j = 0; j < k_Count; j++) {
        input_data[i][j] = static_cast<float>((i + 1) * (j + 2));
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != *std::max_element(input_data[i].begin(), input_data[i].end())) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
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
