#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace goriacheva_k_violation_order_elem_vec {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace goriacheva_k_violation_order_elem_vec
