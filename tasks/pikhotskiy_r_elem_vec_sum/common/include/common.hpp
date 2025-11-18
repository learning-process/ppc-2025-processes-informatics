#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace pikhotskiy_r_elem_vec_sum {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pikhotskiy_r_elem_vec_sum
