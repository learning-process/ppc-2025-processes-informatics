#pragma once

#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"

namespace matrix_band_multiplication {

class MatrixBandMultiplicationMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MatrixBandMultiplicationMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_ = 0;
  int world_size_ = 1;
  std::size_t rows_a_ = 0;
  std::size_t cols_a_ = 0;
  std::size_t rows_b_ = 0;
  std::size_t cols_b_ = 0;
  std::vector<double> local_a_;
  std::vector<double> full_b_;
  std::vector<double> local_result_;
  std::vector<int> row_counts_;
  std::vector<int> row_displs_;
  std::vector<int> result_counts_;
  std::vector<int> result_displs_;
};

}  // namespace matrix_band_multiplication
