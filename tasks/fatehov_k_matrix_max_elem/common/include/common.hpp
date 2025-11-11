#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace fatehov_k_matrix_max_elem {

using InType = std::tuple<size_t, size_t, std::vector<double>>;
using OutType = double;
using TestType = std::tuple<int, size_t, size_t, std::vector<double>, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fatehov_k_matrix_max_elem
