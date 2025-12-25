#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterFunctionalTests : public ppc::util::BaseRunFuncTests<InputType, OutputType> {
  InputType test_input_{};
  std::vector<int> send_buffer_;
  std::vector<int> receive_buffer_;

  void PrepareTestData() {
    bool sequential_mode = GetParam().type_of_task == ppc::task::TypeOfTask::kSEQ;

    int process_rank = 0;
    int total_processes = 1;

    if (!sequential_mode) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    }

    const int root_process = 0;
    const int elements_per_process = 50;

    bool is_root = sequential_mode || (process_rank == root_process);

    receive_buffer_.resize(elements_per_process);

    if (is_root) {
      std::size_t total_elements =
          sequential_mode ? elements_per_process : static_cast<std::size_t>(elements_per_process) * total_processes;

      send_buffer_.resize(total_elements);

      for (std::size_t idx = 0; idx < total_elements; ++idx) {
        send_buffer_[idx] = static_cast<int>(idx * 2 + 5);
      }
    }

    test_input_.source_buffer = is_root ? send_buffer_.data() : nullptr;
    test_input_.elements_to_send = elements_per_process;
    test_input_.send_data_type = MPI_INT;
    test_input_.destination_buffer = receive_buffer_.data();
    test_input_.elements_to_receive = elements_per_process;
    test_input_.receive_data_type = MPI_INT;
    test_input_.root_process = root_process;
    test_input_.communicator = MPI_COMM_WORLD;
  }

  bool ValidateOutputData(OutputType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    bool sequential_mode = GetParam().type_of_task == ppc::task::TypeOfTask::kSEQ;
    int process_rank = 0;

    if (!sequential_mode) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }

    const int elements_per_process = 50;
    std::int64_t global_offset = static_cast<std::int64_t>(process_rank) * elements_per_process;

    const int *result_data = reinterpret_cast<const int *>(output_data.data());
    std::size_t expected_bytes = static_cast<std::size_t>(elements_per_process) * sizeof(int);

    if (output_data.size() != expected_bytes) {
      return false;
    }

    for (int local_idx = 0; local_idx < elements_per_process; ++local_idx) {
      std::int64_t global_idx = global_offset + local_idx;
      int expected_value = static_cast<int>(global_idx * 2 + 5);

      if (result_data[local_idx] != expected_value) {
        return false;
      }
    }

    return true;
  }

  InputType GetInputData() final {
    return test_input_;
  }

 protected:
  void SetUp() override {
    PrepareTestData();
  }
};

TEST_P(PikhotskiyRScatterFunctionalTests, ExecuteScatterFunction) {
  ExecuteTest(GetParam());
}

const auto all_functional_tasks =
    ppc::util::MakeAllFuncTasks<InputType, PikhotskiyRScatterMPI, PikhotskiyRScatterSEQ>();

const auto gtest_parameters = ppc::util::TupleToGTestValues(all_functional_tasks);

INSTANTIATE_TEST_SUITE_P(FunctionalityTests, PikhotskiyRScatterFunctionalTests, gtest_parameters,
                         &PikhotskiyRScatterFunctionalTests::GetTestName);

}  // namespace pikhotskiy_r_scatter
