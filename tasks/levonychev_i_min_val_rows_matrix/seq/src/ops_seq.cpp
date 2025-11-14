#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixSEQ::LevonychevIMinValRowsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LevonychevIMinValRowsMatrixSEQ::ValidationImpl() {
  const size_t vector_size_ = std::get<0>(GetInput()).size();
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());
  if (vector_size_ == 0 || ROWS == 0 || COLS == 0) {
    return false;
  }
  if (vector_size_ != static_cast<size_t>(ROWS * COLS)) {
    return false;
  }
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PreProcessingImpl() {
  GetOutput().resize(static_cast<size_t>(std::get<1>(GetInput())));
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::RunImpl() {
  const std::vector<int> &matrix = std::get<0>(GetInput());
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());
  OutType& result = GetOutput();
  
  for (int i = 0; i < ROWS; ++i) {
    int min_val = matrix[COLS * i];
    for (int j = 1; j < COLS; ++j) {
      if (matrix[COLS * i + j] < min_val) {
        min_val = matrix[COLS * i + j];
      }
    }
    result[i] = min_val;
  }

  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PostProcessingImpl() {
  return GetOutput().size() == std::get<1>(GetInput());
}

}  // namespace levonychev_i_min_val_rows_matrix
