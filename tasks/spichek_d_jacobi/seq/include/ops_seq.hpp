#pragma once

#include "spichek_d_jacobi/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_jacobi {

class SpichekDJacobiSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SpichekDJacobiSEQ(InType task_data);

 private:
  InType input_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_jacobi
