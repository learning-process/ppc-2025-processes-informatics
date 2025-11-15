#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace makovskiy_i_min_value_in_matrix_rows {

MinValueSEQ::MinValueSEQ(const InType &in) {
  InType temp(in);
  this->GetOutput() = {};
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool MinValueSEQ::ValidationImpl() {
  const auto &mat = this->GetInput();
  if (mat.empty()) {
    return false;
  }
  for (const auto &row : mat) {
    if (row.empty()) {
      return false;
    }
  }
  return true;
}

bool MinValueSEQ::PreProcessingImpl() {
  const auto &mat = this->GetInput();
  if (!mat.empty()) {
    this->GetOutput().resize(mat.size(), 0);
  }
  return true;
}

bool MinValueSEQ::RunImpl() {
  const auto &mat = this->GetInput();
  auto &out = this->GetOutput();

  for (const auto &row : mat) {
    if (row.empty()) {
      out.clear();
      return true;
    }
  }

  for (std::size_t i = 0; i < mat.size(); ++i) {
    const auto &row = mat[i];
    int minv = row[0];
    for (std::size_t j = 1; j < row.size(); ++j) {
      minv = std::min(minv, row[j]);
    }
    out[i] = minv;
  }
  return true;
}

bool MinValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_min_value_in_matrix_rows
