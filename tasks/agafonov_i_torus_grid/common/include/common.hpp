#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace agafonov_i_torus_grid {

struct TorusTaskData {
  int value;
  int source_rank;
  int dest_rank;
};

using InType = TorusTaskData;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace agafonov_i_torus_grid
