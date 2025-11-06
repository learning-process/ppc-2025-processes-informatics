#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace pylaeva_s_max_elem_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pylaeva_s_max_elem_matrix
