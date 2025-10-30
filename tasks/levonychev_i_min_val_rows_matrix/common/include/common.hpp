#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace levonychev_i_min_val_rows_matrix {

using InType = std::tuple<std::vector<double>, int, int>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_min_val_rows_matrix
