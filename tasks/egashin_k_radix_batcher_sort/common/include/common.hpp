#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace egashin_k_radix_batcher_sort {

using InType = std::vector<double>;
using OutType = std::vector<double>;

using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egashin_k_radix_batcher_sort
