#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kurpiakov_a_elem_vec_sum {
constexpr double kEps = 10e-12;
using InType = std::tuple<int, std::vector<double>>;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kurpiakov_a_elem_vec_sum
