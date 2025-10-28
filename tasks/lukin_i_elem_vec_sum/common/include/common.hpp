#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace lukin_i_elem_vec_sum {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lukin_i_elem_vec_sum
