#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace volkov_a_count_word_line {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace volkov_a_count_word_line
