#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lobanov_d_multiply_matrix_ccs {

struct CompressedColumnMatrix {
  int row_count = 0;
  int column_count = 0;
  int non_zero_count = 0;
  std::vector<double> value_data;
  std::vector<int> row_index_data;
  std::vector<int> column_pointer_data;

  CompressedColumnMatrix() {
    memset(this, 0, sizeof(CompressedColumnMatrix));
    row_count = 0;
    column_count = 0;
    non_zero_count = 0;
  }

  CompressedColumnMatrix(const CompressedColumnMatrix &other) {
    memset(this, 0, sizeof(CompressedColumnMatrix));
    row_count = other.row_count;
    column_count = other.column_count;
    non_zero_count = other.non_zero_count;
    value_data = other.value_data;
    row_index_data = other.row_index_data;
    column_pointer_data = other.column_pointer_data;
  }

  CompressedColumnMatrix &operator=(const CompressedColumnMatrix &other) {
    if (this != &other) {
      row_count = other.row_count;
      column_count = other.column_count;
      non_zero_count = other.non_zero_count;
      value_data = other.value_data;
      row_index_data = other.row_index_data;
      column_pointer_data = other.column_pointer_data;
    }
    return *this;
  }

  ~CompressedColumnMatrix() = default;

  bool IsValid() const {
    if (row_count < 0 || column_count < 0 || non_zero_count < 0) {
      return false;
    }
    if (non_zero_count > 0) {
      if (value_data.size() != static_cast<size_t>(non_zero_count)) {
        return false;
      }
      if (row_index_data.size() != static_cast<size_t>(non_zero_count)) {
        return false;
      }
    }
    if (column_pointer_data.size() != static_cast<size_t>(column_count + 1)) {
      return false;
    }
    return true;
  }
};

using InType = std::pair<CompressedColumnMatrix, CompressedColumnMatrix>;
using OutType = CompressedColumnMatrix;
using TestType = std::tuple<std::string, CompressedColumnMatrix, CompressedColumnMatrix, CompressedColumnMatrix>;
using BaseTask = ppc::task::Task<InType, OutType>;

CompressedColumnMatrix CreateRandomCompressedColumnMatrix(int row_count, int column_count, double density_factor,
                                                          int seed);

}  // namespace lobanov_d_multiply_matrix_ccs
