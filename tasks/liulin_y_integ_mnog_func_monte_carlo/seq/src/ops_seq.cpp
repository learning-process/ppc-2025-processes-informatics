#include "liulin_y_integ_mnog_func_monte_carlo/seq/include/ops_seq.hpp"

#include <cstddef>
#include <random>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

LiulinYIntegMnogFuncMonteCarloSEQ::LiulinYIntegMnogFuncMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.num_points > 0 && input.x_min <= input.x_max && input.y_min <= input.y_max;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PreProcessingImpl() {
  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  if (input.num_points <= 0) {
    result = 0.0;
    return true;
  }

  const double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist_x(input.x_min, input.x_max);
  std::uniform_real_distribution<double> dist_y(input.y_min, input.y_max);

  double sum = 0.0;
  for (long long i = 0; i < input.num_points; ++i) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    sum += input.f(x, y);
  }

  result = sum / input.num_points * area;

  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_integ_mnog_func_monte_carlo
