#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dolov_v_torus_topology
