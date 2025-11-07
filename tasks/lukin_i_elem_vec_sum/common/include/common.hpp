#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lukin_i_elem_vec_sum {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<size_t, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lukin_i_elem_vec_sum
