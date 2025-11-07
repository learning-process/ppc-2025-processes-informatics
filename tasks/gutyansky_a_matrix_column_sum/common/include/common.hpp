#pragma once

#include <string>
#include <tuple>

#include "matrix.hpp"
#include "task/include/task.hpp"
#include "vector.hpp"

namespace gutyansky_a_matrix_column_sum {

using InType = Matrix;
using OutType = Vector;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gutyansky_a_matrix_column_sum
