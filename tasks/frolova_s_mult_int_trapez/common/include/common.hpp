#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace frolova_s_star_topology {

struct TrapezoidalIntegrationInput {
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> number_of_intervals;
  double (*function)(std::vector<double>);
};

using InType = TrapezoidalIntegrationInput;
using OutType = double;
using TestType = std::tuple<TrapezoidalIntegrationInput, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace frolova_s_star_topology
