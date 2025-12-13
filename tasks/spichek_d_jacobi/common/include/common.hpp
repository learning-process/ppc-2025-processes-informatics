#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace spichek_d_jacobi {

using Matrix = std::vector<std::vector<double>>;

using Vector = std::vector<double>;

using InType = std::tuple<Matrix, Vector, double, int>;

using OutType = Vector;

using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_jacobi
