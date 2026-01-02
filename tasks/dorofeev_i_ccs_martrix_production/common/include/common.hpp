#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace dorofeev_i_ccs_martrix_production {

struct CCSMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<double> values;    // ненулевые элементы
  std::vector<int> row_indices;  // индексы строк
  std::vector<int> col_ptr;      // указатели начала столбцов
};

using InType = std::pair<CCSMatrix, CCSMatrix>;  // A, B
using OutType = CCSMatrix;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dorofeev_i_ccs_martrix_production
