#pragma once

#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"

namespace nikitina_v_max_elem_matr {

class MaxElementMatrMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MaxElementMatrMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows, cols;
  int global_max;
  std::vector<int> matrix_;
};

}  // namespace nikitina_v_max_elem_matr
