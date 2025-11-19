#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_matrix_column_sum {

RozenbergAMatrixColumnSumSEQ::RozenbergAMatrixColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RozenbergAMatrixColumnSumSEQ::ValidationImpl() {
  bool rows_empty = false;
  for (size_t i = 0; i < GetInput().size(); i++) {
    if (GetInput()[i].empty()) {
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

  for (size_t i = 0; i < GetInput().size(); i++) { 
    for (size_t j = 0; j < GetInput()[i].size(); j++) {
      GetOutput()[j] += GetInput()[i][j];
    }
  }

  return !(GetOutput().empty());
}

bool RozenbergAMatrixColumnSumSEQ::PostProcessingImpl() {
  return !(GetOutput().empty());
}

}  // namespace rozenberg_a_matrix_column_sum
