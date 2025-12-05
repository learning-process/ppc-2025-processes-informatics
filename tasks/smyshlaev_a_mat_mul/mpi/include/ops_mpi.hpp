#pragma once

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_mat_mul {

class SmyshlaevAMatMulMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SmyshlaevAMatMulMPI(const InType &in);

 private:
  std::vector<double> mat_b_transposed;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_mat_mul
