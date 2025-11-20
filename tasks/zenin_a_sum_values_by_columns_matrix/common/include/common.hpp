#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

using InType = std::tuple<size_t, size_t, std::vector<double>>;  // rows -> columns -> size
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zenin_a_sum_values_by_columns_matrix
