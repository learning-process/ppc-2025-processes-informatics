#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nikolaev_d_sparse_matrix_mult_crs_double {

using InType = int;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct MatrixCRS {
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_pointers;
  int rows = 0;
  int cols = 0;

  MatrixCRS() = default;
};

}  // namespace nikolaev_d_sparse_matrix_mult_crs_double
