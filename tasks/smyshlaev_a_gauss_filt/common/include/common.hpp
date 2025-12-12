#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace smyshlaev_a_gauss_filt {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_gauss_filt
