#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace yurkin_counting_number {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace yurkin_counting_number
