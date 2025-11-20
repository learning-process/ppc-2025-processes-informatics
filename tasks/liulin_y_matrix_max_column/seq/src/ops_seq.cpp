#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

LiulinYMatrixMaxColumnSEQ::LiulinYMatrixMaxColumnSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LiulinYMatrixMaxColumnSEQ::ValidationImpl() {
  const auto& in = GetInput();

  if (in.empty() || in[0].empty())
    return false;

  size_t cols = in[0].size();
  for (const auto& row : in)
    if (row.size() != cols)
      return false;

  return GetOutput().empty();
}

bool LiulinYMatrixMaxColumnSEQ::PreProcessingImpl() {
  const size_t cols = GetInput()[0].size();
  GetOutput().assign(cols, std::numeric_limits<int>::min());
  return true;
}

bool LiulinYMatrixMaxColumnSEQ::RunImpl() {
  const auto& matrix = GetInput();
  auto& out = GetOutput();

  const int rows = matrix.size();
  const int cols = matrix[0].size();

  for (int col = 0; col < cols; col++) {
    std::vector<int> column(rows);
    for (int r = 0; r < rows; r++) {
      column[r] = matrix[r][col];
    }

    int size = rows;
    std::vector<int> temp = column;

    while (size > 1) {
      int new_size = 0;
      for (int i = 0; i < size; i += 2) {
        if (i + 1 < size)
          temp[new_size] = std::max(temp[i], temp[i + 1]);
        else
          temp[new_size] = temp[i];

        new_size++;
      }
      size = new_size;
    }

    out[col] = temp[0];
  }

  return true;
}

bool LiulinYMatrixMaxColumnSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_matrix_max_column
