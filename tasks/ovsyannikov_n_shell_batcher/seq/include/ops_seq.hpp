#pragma once

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit OvsyannikovNShellBatcherSEQ(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ShellSort(std::vector<int> &arr);
};

}  // namespace ovsyannikov_n_shell_batcher
