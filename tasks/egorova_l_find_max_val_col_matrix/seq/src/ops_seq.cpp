#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace egorova_l_find_max_val_col_matrix {

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

EgorovaLFindMaxValColMatrixSEQ::EgorovaLFindMaxValColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

bool EgorovaLFindMaxValColMatrixSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool EgorovaLFindMaxValColMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool EgorovaLFindMaxValColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();
  std::vector<int> result(cols, std::numeric_limits<int>::min());

  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      if (matrix[i][j] > result[j]) {
        result[j] = matrix[i][j];
      }
    }
  }

  GetOutput() = result;
  return true;
}

bool EgorovaLFindMaxValColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
