#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace frolova_s_sum_elem_matrix {

using Matrix = std::vector<std::vector<int>>;
using InType = Matrix;
using OutType = int64_t;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace frolova_s_sum_elem_matrix
