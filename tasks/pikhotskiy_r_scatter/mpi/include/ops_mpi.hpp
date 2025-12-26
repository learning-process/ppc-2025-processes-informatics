#pragma once

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterMPI : public TaskBase {
 public:
  static constexpr ppc::task::TypeOfTask GetTaskType() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit PikhotskiyRScatterMPI(const InputType &input_args);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int mpi_rank_{0};              // Инициализация по умолчанию
  int mpi_size_{0};              // Инициализация по умолчанию
  bool is_root_process_{false};  // Инициализация по умолчанию
};

}  // namespace pikhotskiy_r_scatter
