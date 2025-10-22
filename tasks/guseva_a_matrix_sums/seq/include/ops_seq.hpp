#pragma once

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "task/include/task.hpp"

namespace guseva_a_matrix_sums {

class GusevaAMatrixSumsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GusevaAMatrixSumsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  static bool PreProcessingImpl() override;
  static bool RunImpl() override;
  static bool PostProcessingImpl() override;
};

}  // namespace guseva_a_matrix_sums
