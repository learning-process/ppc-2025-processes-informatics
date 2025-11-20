#pragma once

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "task/include/task.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LiulinYMatrixMaxColumnMPI(const InType &in);

 private:
  static int tournament_max(const std::vector<int> &column);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace liulin_y_matrix_max_column
