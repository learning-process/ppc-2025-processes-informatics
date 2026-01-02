#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"
namespace shvetsova_k_rad_sort_batch_merge {
using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;
}  // namespace shvetsova_k_rad_sort_batch_merge
