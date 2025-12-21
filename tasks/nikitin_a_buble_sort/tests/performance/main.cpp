#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"
#include "nikitin_a_buble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_buble_sort {

class NikitinABubleSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int kVectorSize_ = 10000;  // 10000 элементов как в условии
  InType input_data_;

  void SetUp() override {
    // Для perf-тестов всегда используем случайные данные
    std::vector<double> vector_data(kVectorSize_);
    
    // Генерируем случайные данные
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);
    
    for (int i = 0; i < kVectorSize_; ++i) {
      vector_data[i] = dist(gen);
    }
    
    input_data_ = vector_data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем размер
    if (output_data.size() != input_data_.size()) {
      return false;
    }
    
    // Проверяем, что массив отсортирован по возрастанию
    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i - 1] > output_data[i]) {
        return false;
      }
    }
    
    // Проверяем, что все элементы из входного массива присутствуют в выходном
    std::vector<double> sorted_input = input_data_;
    std::sort(sorted_input.begin(), sorted_input.end());
    
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - sorted_input[i]) > 1e-10) {
        return false;
      }
    }
    
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinABubleSortPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

// Создаем perf задачи
const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NikitinABubleSortMPI, NikitinABubleSortSEQ>(
    PPC_SETTINGS_nikitin_a_buble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinABubleSortPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinABubleSortPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_buble_sort
