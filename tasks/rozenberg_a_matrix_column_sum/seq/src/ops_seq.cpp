#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"

namespace rozenberg_a_matrix_column_sum {

RozenbergAMatrixColumnSumSEQ::RozenbergAMatrixColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RozenbergAMatrixColumnSumSEQ::ValidationImpl() {
  bool rows_empty = false;
  for (const auto &i : GetInput()) {
    if (i.empty()) {
      rows_empty = true;
      break;
    }
  }
  return (!(GetInput().empty())) && (GetOutput().empty()) && (!rows_empty);
}

bool RozenbergAMatrixColumnSumSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput()[0].size());
  return GetOutput().size() == GetInput()[0].size();
}

bool RozenbergAMatrixColumnSumSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  std::fill(GetOutput().begin(), GetOutput().end(), 0);

  for (auto &i : GetInput()) {
    for (size_t j = 0; j < i.size(); j++) {
      GetOutput()[j] += i[j];
    }
  }

  return !(GetOutput().empty());
}

bool RozenbergAMatrixColumnSumSEQ::PostProcessingImpl() {
  return !(GetOutput().empty());
}

}  // namespace rozenberg_a_matrix_column_sum
