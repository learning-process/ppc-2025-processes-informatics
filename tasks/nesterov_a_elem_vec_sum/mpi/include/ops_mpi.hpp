#pragma once

#include "nesterov_a_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nesterov_a_elem_vec_sum {

class NesterovAElemVecSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NesterovAElemVecSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nesterov_a_elem_vec_sum
