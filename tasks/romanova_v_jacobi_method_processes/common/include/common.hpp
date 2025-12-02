#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace romanova_v_jacobi_method_processes {

using InType = std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>, double, size_t>; //x, A, b, eps, iterations
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanova_v_jacobi_method_processes
