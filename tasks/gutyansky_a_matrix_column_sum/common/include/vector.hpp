#pragma once

#include <cstddef>
#include <vector>

namespace gutyansky_a_matrix_column_sum {

struct Vector {
  size_t size;
  std::vector<int64_t> data;

  friend bool operator==(const Vector &v_left, const Vector &v_right) {
    return v_left.size == v_right.size && v_left.data == v_right.data;
  }
};

}  // namespace gutyansky_a_matrix_column_sum
