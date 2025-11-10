#pragma once

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKMatrixMaxElemMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FatehovKMatrixMaxElemMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int rows = 0;
  int columns = 0;
  std::vector<double> matrix = {};
  static const int MAX_ROWS = 10000;
  static const int MAX_COLS = 10000;
  static const int MAX_MATRIX_SIZE = 1000000;
  double max = 0.0;
};

}  // namespace fatehov_k_matrix_max_elem
