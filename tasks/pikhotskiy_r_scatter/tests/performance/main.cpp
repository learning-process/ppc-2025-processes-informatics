// NOLINTBEGIN
#ifndef PIKHOTSKIY_R_SCATTER_PERF_TESTS_HPP
#define PIKHOTSKIY_R_SCATTER_PERF_TESTS_HPP

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterPerfTests : public ppc::util::BaseRunPerfTests<InputType, OutputType> {
  InputType test_input_{};
  std::vector<int> send_data_{};
  std::vector<int> receive_data_{};

  static constexpr int kElementsPerProcess = 10000000;

  static bool IsSequentialTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();
    return test_name.find("sequential") != std::string::npos || test_name.find("seq") != std::string::npos;
  }

  // NOLINTNEXTLINE
  void SetUp() override {
    bool sequential_mode = IsSequentialTest();

    int process_rank = 0;
    int total_processes = 1;

    if (!sequential_mode) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    }

    const int root_process = 0;
    bool is_root = sequential_mode || (process_rank == root_process);

    // Подготовка приемного буфера
    receive_data_.resize(kElementsPerProcess);

    // Подготовка отправляемых данных (только на корневом процессе)
    if (is_root) {
      std::size_t total_send_elements =
          sequential_mode ? kElementsPerProcess : static_cast<std::size_t>(kElementsPerProcess) * total_processes;

      send_data_.resize(total_send_elements);

      // Заполняем тестовыми данными: значение = индекс * 7 + 3
      for (std::size_t idx = 0; idx < total_send_elements; ++idx) {
        send_data_[idx] = static_cast<int>((idx * 7) + 3);
      }
    }

    const void *send_buffer_ptr = nullptr;
    if (is_root) {
      send_buffer_ptr = send_data_.data();
    }

    test_input_.source_buffer = send_buffer_ptr;
    test_input_.elements_to_send = kElementsPerProcess;
    test_input_.send_data_type = MPI_INT;
    test_input_.destination_buffer = receive_data_.data();
    test_input_.elements_to_receive = kElementsPerProcess;
    test_input_.receive_data_type = MPI_INT;
    test_input_.root_process = root_process;
    test_input_.communicator = MPI_COMM_WORLD;
  }

  // NOLINTNEXTLINE
  bool CheckTestOutputData(OutputType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    bool sequential_mode = IsSequentialTest();
    int process_rank = 0;

    if (!sequential_mode) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }

    if (test_input_.send_data_type == MPI_INT) {
      std::int64_t global_start_index = static_cast<std::int64_t>(process_rank) * kElementsPerProcess;

      const int *received_ints = reinterpret_cast<const int *>(output_data.data());
      std::size_t expected_size = static_cast<std::size_t>(kElementsPerProcess) * sizeof(int);

      if (output_data.size() != expected_size) {
        return false;
      }

      // Проверяем корректность полученных данных
      for (int local_idx = 0; local_idx < kElementsPerProcess; ++local_idx) {
        std::int64_t global_idx = global_start_index + local_idx;
        int expected_value = static_cast<int>((global_idx * 7) + 3);

        if (received_ints[local_idx] != expected_value) {
          return false;
        }
      }
    } else {
      return false;
    }

    return true;
  }

  InputType GetTestInputData() final {
    return test_input_;
  }

  // Добавляем свою функцию для генерации уникальных имен тестов
  static std::string CustomPerfTestName(const testing::TestParamInfo<PerfTask> &test_param_info) {
    const auto &perf_task = test_param_info.param;
    std::string name = perf_task.task_name;

    // Заменяем все неалфавитно-цифровые символы на '_'
    std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');

    // Добавляем суффикс для уникальности
    return "pipeline_pikhotskiy_r_scatter_" + name;
  }
};

namespace {

// NOLINTNEXTLINE
TEST_P(PikhotskiyRScatterPerfTests, RunPerformanceTests) {
  ExecuteTest(GetParam());
}

// Используем алиасы для типов задач с правильным именованием
using MpiTask = PikhotskiyRScatterMPI;
using SeqTask = PikhotskiyRScatterSEQ;

// NOLINTNEXTLINE
const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InputType, MpiTask, SeqTask>(PPC_SETTINGS_pikhotskiy_r_scatter);

// NOLINTNEXTLINE
const auto kGtestParamValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// Используем нашу кастомную функцию для именования тестов
// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(RunModeTests, PikhotskiyRScatterPerfTests, kGtestParamValues,
                         PikhotskiyRScatterPerfTests::CustomPerfTestName);

}  // namespace
}  // namespace pikhotskiy_r_scatter

#endif  // PIKHOTSKIY_R_SCATTER_PERF_TESTS_HPP
        // NOLINTEND
