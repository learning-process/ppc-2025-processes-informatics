#include "liulin_y_vert_strip_diag_matrix_vect_mult/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "liulin_y_vert_strip_diag_matrix_vect_mult/common/include/common.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

LiulinYVertStripDiagMatrixVectMultSEQ::LiulinYVertStripDiagMatrixVectMultSEQ(const InType &in) {
  GetInput() = InType{};
  GetInput() = in;
  GetOutput().clear();
}

bool LiulinYVertStripDiagMatrixVectMultSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);
  const auto &vect = std::get<1>(input);

  if (matrix.empty() && vect.empty()) {
    return true;
  }

  return true;
}

bool LiulinYVertStripDiagMatrixVectMultSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput().clear();
    return true;
  }

  const std::size_t rows = matrix.size();
  GetOutput().assign(rows, 0);
  return true;
}

bool LiulinYVertStripDiagMatrixVectMultSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);
  const auto &vect = std::get<1>(input);
  auto &out = GetOutput();

  if (matrix.empty() || matrix[0].empty()) {
    return true;
  }

  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[i] += matrix[i][j] * vect[j];
    }
  }

  return true;
}

bool LiulinYVertStripDiagMatrixVectMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
