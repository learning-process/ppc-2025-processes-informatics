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

  CompressedColumnMatrix() : row_count(0), column_count(0), non_zero_count(0) {}
  CompressedColumnMatrix(int r, int c, int nz) : row_count(r), column_count(c), non_zero_count(nz) {}
};

using InType = std::pair<CompressedColumnMatrix, CompressedColumnMatrix>;
using OutType = CompressedColumnMatrix;
using TestType = std::tuple<std::string, CompressedColumnMatrix, CompressedColumnMatrix, CompressedColumnMatrix>;
using BaseTask = ppc::task::Task<InType, OutType>;

CompressedColumnMatrix CreateRandomCompressedColumnMatrix(int row_count, int column_count, double density_factor,
                                                          int seed);

}  // namespace lobanov_d_multiply_matrix_ccs
