#pragma once

#include <mpi.h>

#include <stdexcept>
#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit OvsyannikovNShellBatcherMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ShellSort(std::vector<int> &arr);
  bool need_finalize_ = false;
};

}  // namespace ovsyannikov_n_shell_batcher
