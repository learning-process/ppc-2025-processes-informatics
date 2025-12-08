#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace matrix_band_multiplication {

// GCOVR_EXCL_START
// LCOV_EXCL_START
struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> values;
};

struct MatrixMulInput {
  Matrix a;
  Matrix b;
};
// GCOVR_EXCL_STOP
// LCOV_EXCL_STOP

using InType = MatrixMulInput;
using OutType = Matrix;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::size_t FlattenIndex(std::size_t row, std::size_t col, std::size_t cols) {
  const std::size_t row_offset = row * cols;  // GCOVR_EXCL_LINE LCOV_EXCL_LINE
  return row_offset + col;                    // GCOVR_EXCL_LINE LCOV_EXCL_LINE
}

}  // namespace matrix_band_multiplication
