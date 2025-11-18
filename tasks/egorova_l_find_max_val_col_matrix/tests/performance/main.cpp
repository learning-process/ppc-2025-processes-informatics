#include <gtest/gtest.h>

#include <random>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"
#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 10;  // Очень маленькая матрица для теста
  InType input_data_{};

  void SetUp() override {
    // Создаем простую детерминированную матрицу
    input_data_.resize(kMatrixSize_, std::vector<int>(kMatrixSize_));

    int counter = 1;
    for (int i = 0; i < kMatrixSize_; ++i) {
      for (int j = 0; j < kMatrixSize_; ++j) {
        input_data_[i][j] = counter++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = GetTestInputData();

    if (matrix.empty() || output_data.empty()) {
      return false;
    }

    if (output_data.size() != matrix[0].size()) {
      return false;
    }

    // Вычисляем ожидаемый результат
    std::vector<int> expected(matrix[0].size(), std::numeric_limits<int>::min());
    for (size_t j = 0; j < matrix[0].size(); ++j) {
      for (size_t i = 0; i < matrix.size(); ++i) {
        if (matrix[i][j] > expected[j]) {
          expected[j] = matrix[i][j];
        }
      }
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected[i]) {
        return false;
      }
    }
    return true;
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
