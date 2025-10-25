#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace spichek_d_dot_product_of_vectors {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_dot_product_of_vectors
