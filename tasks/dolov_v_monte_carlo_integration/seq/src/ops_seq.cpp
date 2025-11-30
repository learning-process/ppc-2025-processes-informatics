#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cmath>   // Для std::pow, std::sqrt, std::isfinite
#include <random>  // Для std::mt19937, std::uniform_real_distribution
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "util/include/util.hpp"  // Сохраняем, как в исходном коде

namespace dolov_v_monte_carlo_integration {

DolovVMonteCarloIntegrationSEQ::DolovVMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;  // Инициализация с double
}

bool DolovVMonteCarloIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.func && (in.samples_count > 0) && (in.dimension > 0) &&
         (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
}

bool DolovVMonteCarloIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationSEQ::RunImpl() {
  const auto &in = GetInput();
  const int N = in.samples_count;
  const int D = in.dimension;
  const double R = in.radius;
  const double R_sq = R * R;

  // Инициализация генератора случайных чисел
  std::mt19937 generator(42);
  // Единое распределение для диапазона [-R, R]
  std::uniform_real_distribution<double> distribution(-R, R);

  double sum = 0.0;
  std::vector<double> point(D);

  for (int sample = 0; sample < N; ++sample) {
    double distance_sq = 0.0;
    bool is_in_domain = true;

    // Генерация случайной точки в гиперкубе: [center[d] - R, center[d] + R]
    for (int d = 0; d < D; ++d) {
      point[d] = in.center[d] + distribution(generator);
    }

    if (in.domain_type == IntegrationDomain::kHyperSphere) {
      // Проверка попадания в гиперсферу (расчет L2-нормы)
      for (int d = 0; d < D; ++d) {
        distance_sq += std::pow(point[d] - in.center[d], 2);
      }

      if (distance_sq > R_sq) {
        is_in_domain = false;
      }
    }

    // Суммирование
    if (is_in_domain) {
      sum += in.func(point);
    }
  }

  // Объем описанного гиперкуба V_cube = (2*R)^D
  const double V_cube = std::pow(2.0 * R, D);

  // Финальный расчет: Integral = V_cube * (среднее_значение_функции в кубе)
  double integral_approximation = V_cube * (sum / N);

  GetOutput() = integral_approximation;

  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return true;  // Не требуются действия после обработки
}

}  // namespace dolov_v_monte_carlo_integration
