#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace fatehov_k_matrix_max_elem {

using InType = std::tuple<int, int, std::vector<double>>;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fatehov_k_matrix_max_elem
