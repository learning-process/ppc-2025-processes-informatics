#pragma once

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"
#include <vector>

namespace goriacheva_k_violation_order_elem_vec {

class GoriachevaKViolationOrderElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GoriachevaKViolationOrderElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int>input_vec;
  int result = 0;
};

}  // namespace goriacheva_k_violation_order_elem_vec
