#pragma once

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

class KosolapovVMaxValuesInColMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KosolapovVMaxValuesInColMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> CalculateLocalMax(const std::vector<std::vector<int>>& matrix,int rank, int processes_count, int columns);
};

}  // namespace kosolapov_v_max_values_in_col_matrix
