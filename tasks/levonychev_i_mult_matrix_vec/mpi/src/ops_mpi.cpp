#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

LevonychevIMultMatrixVecMPI::LevonychevIMultMatrixVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LevonychevIMultMatrixVecMPI::ValidationImpl() {
  const size_t matrix_size = std::get<0>(GetInput()).size();
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  bool is_correct_matrix_size = (matrix_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
  bool is_correct_vector_size = (static_cast<size_t>(cols) == std::get<3>(GetInput()).size());
  return matrix_size != 0 && rows != 0 && cols != 0 && is_correct_matrix_size && is_correct_vector_size;
}

bool LevonychevIMultMatrixVecMPI::PreProcessingImpl() {
  GetOutput().resize(static_cast<size_t>(std::get<1>(GetInput())));
  return true;
}

bool LevonychevIMultMatrixVecMPI::RunImpl() {
  const std::vector<int>& matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  const std::vector<int>& vec_x = std::get<3>(GetInput());

  OutType &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    int scalar_product = 0;
    for (int j = 0; j < cols; ++j) {
      scalar_product += matrix[i*cols + j] * vec_x[j];
    }
    result[i] = scalar_product;
  }
  return true;
}

bool LevonychevIMultMatrixVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_mult_matrix_vec
