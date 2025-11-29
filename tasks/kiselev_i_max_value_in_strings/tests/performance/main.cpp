#include <gtest/gtest.h>

#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIMaxValueInStringsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int k_count = 10000;     // количество строк
  const int k_max_cols = 10000;  // макс длина строки
 protected:
  void SetUp() override {
    // big matrix
    input_data_.resize(k_count);
    for (int i = 0; i < k_count; ++i) {
      int len_row = 1 + (int)(i % k_max_cols);
      input_data_[i].resize(len_row);
      for (int j = 0; j < len_row; ++j) {
        input_data_[i][j] = (i + 2 + j + 7) % 100;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для performance тестов проверяем, что результат не пустой и правильного размера
    return !output_data.empty() && output_data.size() == input_data_.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KiselevIMaxValueInStringsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(
    PPC_SETTINGS_kiselev_i_max_value_in_strings);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIMaxValueInStringsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIMaxValueInStringsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_max_value_in_strings
