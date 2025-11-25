#pragma once

#include "task/include/task.hpp"

#include <string>
#include <tuple>
#include <vector>

namespace liulin_y_matrix_max_column {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_matrix_max_column
