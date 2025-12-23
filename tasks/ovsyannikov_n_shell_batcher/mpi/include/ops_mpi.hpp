#pragma once

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

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ShellSort(std::vector<int> &arr);
};

}  // namespace ovsyannikov_n_shell_batcher