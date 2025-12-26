#pragma once

#include <ostream>  // добавьте для оператора вывода
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

  CompressedColumnMatrix() : row_count(0), column_count(0), non_zero_count(0) {}

  CompressedColumnMatrix(int r, int c, int nz) : row_count(r), column_count(c), non_zero_count(nz) {
    if (nz > 0) {
      value_data.reserve(nz);
      row_index_data.reserve(nz);
    }
    column_pointer_data.reserve(c + 1);
  }

  CompressedColumnMatrix(const CompressedColumnMatrix &other) = default;

  CompressedColumnMatrix &operator=(const CompressedColumnMatrix &other) = default;

  ~CompressedColumnMatrix() = default;

  void ZeroInitialize() {
    row_count = 0;
    column_count = 0;
    non_zero_count = 0;
    value_data.clear();
    row_index_data.clear();
    column_pointer_data.clear();
  }

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
    if (!column_pointer_data.empty() && column_pointer_data[0] != 0) {
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
