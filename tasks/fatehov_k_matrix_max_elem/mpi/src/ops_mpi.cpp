#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"

#include <numeric>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace fatehov_k_matrix_max_elem {

FatehovKMatrixMaxElemMPI::FatehovKMatrixMaxElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKMatrixMaxElemMPI::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= MAX_ROWS) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= MAX_COLS) &&
         (std::get<0>(data) * std::get<1>(data) <= MAX_MATRIX_SIZE) &&
         (std::get<2>(data).size() <= MAX_MATRIX_SIZE &&
          std::get<2>(data).size() == std::get<0>(data) * std::get<1>(data)) &&
         (!std::get<2>(data).empty());
}

bool FatehovKMatrixMaxElemMPI::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixMaxElemMPI::RunImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t columns = std::get<1>(data);
  std::vector<double> matrix = std::get<2>(data);
  double max = matrix[0];
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      if (matrix[i * columns + j] > max) {
        max = matrix[i * columns + j];
      }
    }
  }
  GetOutput() = max;
  return true;
}

bool FatehovKMatrixMaxElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_max_elem
