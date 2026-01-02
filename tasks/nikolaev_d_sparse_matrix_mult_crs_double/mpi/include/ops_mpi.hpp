#pragma once

#include <string>
#include <vector>

#include "nikolaev_d_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikolaev_d_sparse_matrix_mult_crs_double {

class NikolaevDSparseMatrixMultCrsDoubleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikolaevDSparseMatrixMultCrsDoubleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastMatrixInfo();
  void DistributeMatrixA();
  void DistributeMatrixB();
  void GatherResults();

  MatrixCRS matrix_A_;
  MatrixCRS matrix_B_;
  MatrixCRS result_matrix_;
  MatrixCRS local_a_rows_;
  MatrixCRS local_result_;

  std::string matrix_A_filename_;
  std::string matrix_B_filename_;
  std::vector<int> local_rows_;

  int rows_A_ = 0;
  int cols_A_ = 0;
  int rows_B_ = 0;
  int cols_B_ = 0;
};

}  // namespace nikolaev_d_sparse_matrix_mult_crs_double
