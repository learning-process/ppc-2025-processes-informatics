#pragma once

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterSEQ : public TaskBase {
 public:
  static constexpr ppc::task::TypeOfTask GetTaskType() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit PikhotskiyRScatterSEQ(const InputType &input_args);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pikhotskiy_r_scatter
