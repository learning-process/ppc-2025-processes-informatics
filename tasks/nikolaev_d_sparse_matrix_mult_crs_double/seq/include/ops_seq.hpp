#pragma once

#include <string>

#include "nikolaev_d_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikolaev_d_sparse_matrix_mult_crs_double {

class NikolaevDSparseMatrixMultCrsDoubleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NikolaevDSparseMatrixMultCrsDoubleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  MatrixCRS matrix_A_;
  MatrixCRS matrix_B_;
  MatrixCRS result_matrix_;
  std::string matrix_A_filename_;
  std::string matrix_B_filename_;
};

}  // namespace nikolaev_d_sparse_matrix_mult_crs_double
