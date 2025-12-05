#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shkrebko_m_hypercube {

struct HypercubeData {
  int value;
  int destination;
  std::vector<int> path;
  bool finish;

  HypercubeData() : value(0), destination(0), finish(false) {}
};

using InType = std::vector<int>;
using OutType = HypercubeData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkrebko_m_hypercube
