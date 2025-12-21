#include "maslova_u_row_matr_vec_mult/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"
#include "util/include/util.hpp"

namespace maslova_u_row_matr_vec_mult {

MaslovaURowMatrVecMultSEQ::MaslovaURowMatrVecMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MaslovaURowMatrVecMultSEQ::ValidationImpl() {
  rreturn (GetInput().first.cols == GetInput().second.size()) && (!GetInput().first.data.empty());
}

bool MaslovaURowMatrVecMultSEQ::PreProcessingImpl() {
  return true;
}

bool MaslovaURowMatrVecMultSEQ::RunImpl() {
  const Matrix& matrix = GetInput().first;
  const std::vector<double>& vector = GetInput().second;
  GetOutput() = std::vector<double>(matrix.rows, 0.0);

  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      GetOutput()[i] += matrix.data[i * matrix.cols + j] * vector[j];
    }
  }
  return true;
}

bool MaslovaURowMatrVecMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_row_matr_vec_mult
