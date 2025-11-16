#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace luzan_e_matrix_rows_sum {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luzan_e_matrix_rows_sum
