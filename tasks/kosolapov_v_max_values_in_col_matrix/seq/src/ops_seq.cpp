#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixSEQ::KosolapovVMaxValuesInColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  for (int i = 0; i < matrix.size()-1; i++)
  {
    if (matrix[i].size() != matrix[i + 1].size())
    {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  for (int i = 0; i < matrix[0].size(); i++)
  {
    int temp_max = matrix[0][i];
    for (int j = 0; j < matrix.size(); j++)
    {
      if (matrix[j][i] > temp_max)
      {
        temp_max = matrix[j][i];
      }
    }
    GetOutput()[i] = temp_max;
  }
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kosolapov_v_max_values_in_col_matrix
