#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixSEQ::LevonychevIMinValRowsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(std::get<1>(in));
}

bool LevonychevIMinValRowsMatrixSEQ::ValidationImpl() {
  if (std::get<0>(GetInput()).size() == 0 || std::get<1>(GetInput()) == 0 || std::get<2>(GetInput()) == 0) {
    return false;
  }
  if (std::get<0>(GetInput()).size() != std::get<1>(GetInput()) * std::get<2>(GetInput())) {
    return false;
  }
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PreProcessingImpl() {
  GetOutput().resize(std::get<1>(GetInput()));
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::RunImpl() {
  const std::vector<double> &matrix = std::get<0>(GetInput());
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());

  for (int i = 0; i < ROWS; ++i) {
    double min_val = matrix[COLS * i];
    for (int j = 1; j < COLS; ++j) {
      if (matrix[COLS * i + j] < min_val) {
        min_val = matrix[COLS * i + j];
      }
    }
    GetOutput()[i] = min_val;
  }

  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PostProcessingImpl() {
  return GetOutput().size() == std::get<1>(GetInput());
}

}  // namespace levonychev_i_min_val_rows_matrix
