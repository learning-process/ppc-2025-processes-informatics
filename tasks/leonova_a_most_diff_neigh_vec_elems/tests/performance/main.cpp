// #include <gtest/gtest.h>

// #include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace leonova_a_most_diff_neigh_vec_elems {

// class MostDiffNeighVecElemsRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const int kCount_ = 100;
//   InType input_data_{};

//   void SetUp() override {
//     //input_data_ = kCount_;
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     //return input_data_ == output_data;
//     return true;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(MostDiffNeighVecElemsRunPerfTestsProcesses, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, LeonovaAMostDiffNeighVecElemsMPI,
//     LeonovaAMostDiffNeighVecElemsSEQ>(PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = MostDiffNeighVecElemsRunPerfTestsProcesses::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, MostDiffNeighVecElemsRunPerfTestsProcesses, kGtestValues, kPerfTestName);

// }  // namespace leonova_a_most_diff_neigh_vec_elems

// #include <gtest/gtest.h>
// #include <vector>
// #include <random>

// #include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace leonova_a_most_diff_neigh_vec_elems {

// class MostDiffNeighVecElemsRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const size_t kVectorSize_ = 1000000; // Большой вектор для performance тестов
//   InType input_data_{};

//   void SetUp() override {
//     // Генерируем большой вектор для performance тестирования
//     input_data_.resize(kVectorSize_);

//     // Заполняем вектор данными, создавая сценарий где максимальная разница
//     // будет в определенном месте для проверки корректности
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int> dist(1, 1000);

//     for (size_t i = 0; i < kVectorSize_; ++i) {
//       input_data_[i] = dist(gen);
//     }

//     // Создаем гарантированную максимальную разницу в середине вектора
//     // для проверки что алгоритм работает корректно
//     if (kVectorSize_ > 100) {
//       input_data_[kVectorSize_ / 2] = 1;
//       input_data_[kVectorSize_ / 2 + 1] = 10000; // Большая разница
//     }
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     // Проверяем что результат логически корректен
//     // В performance тестах полная проверка может быть не нужна,
//     // но базовая валидация важна
//     int diff = std::abs(std::get<0>(output_data) - std::get<1>(output_data));
//     return diff >= 0; // Разница должна быть неотрицательной
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(MostDiffNeighVecElemsRunPerfTestsProcesses, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, LeonovaAMostDiffNeighVecElemsMPI,
//     LeonovaAMostDiffNeighVecElemsSEQ>(PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = MostDiffNeighVecElemsRunPerfTestsProcesses::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, MostDiffNeighVecElemsRunPerfTestsProcesses, kGtestValues, kPerfTestName);

// }  // namespace leonova_a_most_diff_neigh_vec_elems

// #include <gtest/gtest.h>

// #include <random>
// #include <vector>

// #include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace leonova_a_most_diff_neigh_vec_elems {

// class MostDiffNeighVecElemsRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const size_t kVectorSize_ = 1000000;
//   InType input_data_{};
//   OutType expected_output_{};

//   void SetUp() override {
//     input_data_.resize(kVectorSize_);

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int> dist(1, 1000);

//     // Заполняем вектор случайными числами
//     for (size_t i = 0; i < kVectorSize_; ++i) {
//       input_data_[i] = dist(gen);
//     }

//     // Создаем гарантированную МАКСИМАЛЬНУЮ РАЗНИЦУ МЕЖДУ СОСЕДНИМИ ЭЛЕМЕНТАМИ
//     // в определенной позиции
//     size_t max_diff_position = kVectorSize_ / 2;
//     if (max_diff_position + 1 < kVectorSize_) {
//       input_data_[max_diff_position] = 1;
//       input_data_[max_diff_position + 1] = 10000;  // Большая разница между соседями
//       expected_output_ = std::make_tuple(1, 10000);
//     }

//     // Добавляем еще одну большую разницу в другом месте (но меньшую)
//     size_t second_diff_position = kVectorSize_ / 4;
//     if (second_diff_position + 1 < kVectorSize_) {
//       input_data_[second_diff_position] = 100;
//       input_data_[second_diff_position + 1] = 5000;  // Меньшая разница
//     }
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     // Проверяем, что алгоритм нашел именно пару с максимальной разницей
//     // между СОСЕДНИМИ элементами
//     return (output_data == expected_output_) || (std::abs(std::get<0>(output_data) - std::get<1>(output_data)) ==
//     9999);
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(MostDiffNeighVecElemsRunPerfTestsProcesses, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, LeonovaAMostDiffNeighVecElemsMPI, LeonovaAMostDiffNeighVecElemsSEQ>(
//         PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = MostDiffNeighVecElemsRunPerfTestsProcesses::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, MostDiffNeighVecElemsRunPerfTestsProcesses, kGtestValues, kPerfTestName);

// }  // namespace leonova_a_most_diff_neigh_vec_elems

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class MostDiffNeighVecElemsRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kVectorSize_ = 10000000;  // Увеличили в 10 раз
  const int kNumIterations_ = 100;       // Добавили итерации для увеличения времени
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    input_data_.resize(kVectorSize_);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10000);  // Увеличили диапазон

    // Заполняем вектор случайными числами
    for (size_t i = 0; i < kVectorSize_; ++i) {
      input_data_[i] = dist(gen);
    }

    // Создаем несколько гарантированных максимальных разниц в разных местах
    size_t max_diff_position1 = kVectorSize_ / 4;
    size_t max_diff_position2 = kVectorSize_ / 2;
    size_t max_diff_position3 = 3 * kVectorSize_ / 4;

    if (max_diff_position1 + 1 < kVectorSize_) {
      input_data_[max_diff_position1] = 1;
      input_data_[max_diff_position1 + 1] = 50000;  // Большая разница
    }

    if (max_diff_position2 + 1 < kVectorSize_) {
      input_data_[max_diff_position2] = 100;
      input_data_[max_diff_position2 + 1] = 60000;  // Самая большая разница
      expected_output_ = std::make_tuple(100, 60000);
    }

    if (max_diff_position3 + 1 < kVectorSize_) {
      input_data_[max_diff_position3] = 500;
      input_data_[max_diff_position3 + 1] = 45000;  // Большая разница
    }

    // Добавляем паттерны для усложнения вычислений
    for (size_t i = 0; i < kVectorSize_ / 100; i += 10) {
      if (i + 1 < kVectorSize_) {
        input_data_[i] = i % 1000;
        input_data_[i + 1] = (i + 500) % 1000;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что алгоритм нашел пару с достаточно большой разницей
    int diff = std::abs(std::get<0>(output_data) - std::get<1>(output_data));
    return diff >= 50000;  // Ожидаем разницу не менее 50000
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MostDiffNeighVecElemsRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LeonovaAMostDiffNeighVecElemsMPI, LeonovaAMostDiffNeighVecElemsSEQ>(
        PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MostDiffNeighVecElemsRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MostDiffNeighVecElemsRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace leonova_a_most_diff_neigh_vec_elems
