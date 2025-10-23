#pragma once

#include <tuple>

#include "task/include/task.hpp"

namespace guseva_a_matrix_sums {
// InType is tuple of rows, cols and matrix
using InType = std::tuple<uint32_t, uint32_t, std::vector<double>>;
// OutType is std::vector<double> of columns sums
using OutType = std::vector<double>;
using TestType = std::tuple<std::string, InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace guseva_a_matrix_sums
