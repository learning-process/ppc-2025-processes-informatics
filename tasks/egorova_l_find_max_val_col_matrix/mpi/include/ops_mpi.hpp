#pragma once

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLFindMaxValColMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EgorovaLFindMaxValColMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace egorova_l_find_max_val_col_matrix
