#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "frolova_s_star_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_star_topology {

// Определяем типы данных для тестов
using InType = std::vector<int>;
using OutType = bool;  // или другой тип, если нужно

class FrolovaSStarPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kDataLength = 4096;
  InType input_data_{};
  std::vector<int> destinations_{};
  int rank_ = 0;
  int size_ = 0;

  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if (size_ < 3) {
      GTEST_SKIP();
      return;
    }

    // Готовим данные для root процесса
    if (rank_ == 0) {
      destinations_.resize(size_ - 1);
      for (std::size_t i = 0; i < destinations_.size(); ++i) {
        destinations_[i] = static_cast<int>(i + 1);
      }

      std::random_device rd;
      std::shuffle(destinations_.begin(), destinations_.end(), rd);

      // Генерируем входные данные
      input_data_.resize((size_ - 1) * kDataLength);
      std::random_device dev;
      std::mt19937 gen(dev());
      for (size_t i = 0; i < input_data_.size(); ++i) {
        input_data_[i] = static_cast<int>((gen() % 200) - 100);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Здесь должна быть проверка результатов
    // Возвращаем true для примера, нужно адаптировать под вашу логику
    return output_data == true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  // Дополнительный метод для получения destinations
  std::vector<int> GetDestinations() {
    return destinations_;
  }

  // Дополнительный метод для получения ранга
  int GetRank() {
    return rank_;
  }

  // Дополнительный метод для получения размера
  int GetSize() {
    return size_;
  }
};

TEST_P(FrolovaSStarPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

// Создаем все задачи для тестирования
const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSStarTopologyMPI, FrolovaSStarTopologySEQ>(
    PPC_SETTINGS_frolova_s_star_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaSStarPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FrolovaSStarPerfTest, kGtestValues, kPerfTestName);

}  // namespace frolova_s_star_topology
