#pragma once

#include <cstddef>

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luzan_e_matrix_rows_sum {

using InType = std::tuple<std::vector<int>, size_t, size_t>;  // matrix, height, width
using OutType = std::vector<int>;                             // vec of sums, size = height
using TestType = std::tuple<size_t, size_t>;                  // height & width
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luzan_e_matrix_rows_sum
