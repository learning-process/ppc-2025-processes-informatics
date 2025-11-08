#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace gutyansky_a_matrix_column_sum {

struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<int64_t> data;

  friend bool operator==(const Matrix &v_left, const Matrix &v_right) {
    return v_left.rows == v_right.rows && v_left.cols == v_right.cols && v_left.data == v_right.data;
  }
};

}  // namespace gutyansky_a_matrix_column_sum
