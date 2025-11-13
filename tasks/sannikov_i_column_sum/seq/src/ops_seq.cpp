#include "sannikov_i_column_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumSEQ::SannikovIColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &dst = GetInput();
  InType tmp(in);
  dst.swap(tmp);

  GetOutput().clear();
}

bool SannikovIColumnSumSEQ::ValidationImpl() {
  const auto &input_matrix = GetInput();
  return (!input_matrix.empty()) && (input_matrix.front().size() != 0) && (GetOutput().empty());
}

bool SannikovIColumnSumSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().front().size(), 0);
  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::RunImpl() {
  const auto &input_matrix = GetInput();
  if (input_matrix.empty()) {
    return false;
  }

  for (int i = 0; i < (int)input_matrix.size(); i++) {
    for (int j = 0; j < (int)input_matrix[i].size(); j++) {
      GetOutput()[j] += input_matrix[i][j];
    }
  }

  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
