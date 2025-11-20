#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kichanova_k_count_letters_in_str {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kichanova_k_count_letters_in_str
