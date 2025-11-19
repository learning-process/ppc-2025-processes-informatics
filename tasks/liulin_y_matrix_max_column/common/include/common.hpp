#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace liulin_y_matrix_max_column {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_matrix_max_column
