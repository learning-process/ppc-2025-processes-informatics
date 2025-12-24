#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

// Определяем M_PI, если он не определен (для большей совместимости)
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace dolov_v_monte_carlo_integration {

enum class IntegrationDomain { kHyperCube, kHyperSphere };

struct InputParams {
  std::function<double(const std::vector<double> &)> func;
  int dimension = 1;
  int samples_count = 0;
  std::vector<double> center;
  double radius = 1.0;
  IntegrationDomain domain_type = IntegrationDomain::kHyperCube;
};

using InType = InputParams;
using OutType = double;

using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

// Вспомогательные функции для тестирования
inline double func_sum_coords(const std::vector<double> &x) {
  double sum = 0.0;
  for (double val : x) {
    sum += val;
  }
  return sum;
}

inline double func_const_one(const std::vector<double> &) {
  return 1.0;
}

}  // namespace dolov_v_monte_carlo_integration
