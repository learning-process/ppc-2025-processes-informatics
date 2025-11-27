#include <gtest/gtest.h>

#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  // Большой размер для тестирования производительности
  const int kVectorSize_ = 1000000;   // 1 миллион элементов
  OutType expected_result_ = 999999;  // Для знакопеременного вектора

  void SetUp() override {
    // Генерируем большой знакопеременный вектор для тестирования производительности
    std::vector<double> vector_data;
    vector_data.reserve(kVectorSize_);

    int sign_multiplier = 1;
    for (int i = 0; i < kVectorSize_; i++) {
      // Создаем знакопеременную последовательность: 0, -1, 2, -3, 4, -5, ...
      vector_data.push_back(i * sign_multiplier);
      sign_multiplier *= -1;
    }

    input_data_ = vector_data;
    // Для знакопеременного вектора количество чередований = размер - 1
    expected_result_ = kVectorSize_ - 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для MPI процессов с rank != 0 ожидаем значение -1
    if (output_data == -1) {
      return true;
    }

    // Для процесса с rank 0 проверяем корректный результат
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinAVecSignRotationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NikitinAVecSignRotationMPI, NikitinAVecSignRotationSEQ>(
    PPC_SETTINGS_nikitin_a_vec_sign_rotation);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinAVecSignRotationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinAVecSignRotationPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_vec_sign_rotation
