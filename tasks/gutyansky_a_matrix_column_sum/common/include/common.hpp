#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gutyansky_a_matrix_column_sum {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gutyansky_a_matrix_column_sum
