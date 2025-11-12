#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "matrix.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_matrix_column_sum {

using InType = Matrix;
using OutType = std::vector<int64_t>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gutyansky_a_matrix_column_sum
