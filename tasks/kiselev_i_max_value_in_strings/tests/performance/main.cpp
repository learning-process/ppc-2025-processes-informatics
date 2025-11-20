#include <gtest/gtest.h>

#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIMaxValueInStringsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 1000;  // Большая матрица для тестов производительности

 protected:
  void SetUp() override {
    // Создаем большую матрицу для performance тестов
    input_data_.resize(kMatrixSize_);
    for (int i = 0; i < kMatrixSize_; ++i) {
      input_data_[i].resize(kMatrixSize_);
      for (int j = 0; j < kMatrixSize_; ++j) {
        input_data_[i][j] = (i + j) % 100;  // Заполняем значениями
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

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(PPC_SETTINGS_kiselev_i_max_value_in_strings);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIMaxValueInStringsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIMaxValueInStringsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_max_value_in_strings

/*
#include <gtest/gtest.h>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIMaxValueInStringsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    // input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // return input_data_ == output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KiselevIMaxValueInStringsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(PPC_SETTINGS_kiselev_i_max_value_in_strings);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIMaxValueInStringsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIMaxValueInStringsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_max_value_in_strings
*/