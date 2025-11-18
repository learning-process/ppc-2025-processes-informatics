#include "chyokotov_min_val_by_columns/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "util/include/util.hpp"

namespace chyokotov_min_val_by_columns {

ChyokotovMinValByColumnsSEQ::ChyokotovMinValByColumnsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ChyokotovMinValByColumnsSEQ::ValidationImpl() {
  const auto input = GetInput();
  if (input.empty()) {
    return true;
  }

  size_t lengthRow = input[0].size();
  return std::ranges::all_of(input, [lengthRow](const auto &row) { return row.size() == lengthRow; });
}

bool ChyokotovMinValByColumnsSEQ::PreProcessingImpl() {
  const auto input = GetInput();
  if (input.empty()) {
    return true;
  }
  if (!input[0].empty()) {
    GetOutput().resize(input[0].size(), INT_MAX);
  }
  return true;
}

bool ChyokotovMinValByColumnsSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  auto &output = GetOutput();

  for (size_t i = 0; i < matrix[0].size(); i++) {
    for (size_t j = 0; j < matrix.size(); j++) {
      output[i] = std::min(output[i], matrix[j][i]);
    }
  }

  return true;
}

bool ChyokotovMinValByColumnsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_min_val_by_columns
