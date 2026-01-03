#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

struct InputData {
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x0;
  double tolerance{};
  int max_iterations{};
  // если убрать padding {}, Valgrind ругается, если оставить - clang-tidy, поэтому NOLINT
  int padding_{};  // NOLINT(readability-identifier-naming)
};

using InType = InputData;
using OutType = std::vector<double>;

using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egashin_k_iterative_simple
