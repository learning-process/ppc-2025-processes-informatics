#include "telnov_strongin_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "telnov_strongin_algorithm/common/include/common.hpp"

namespace telnov_strongin_algorithm {

TelnovStronginAlgorithmSEQ::TelnovStronginAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovStronginAlgorithmSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return (in.eps > 0.0) && (in.b > in.a);
}

bool TelnovStronginAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool TelnovStronginAlgorithmSEQ::RunImpl() {
  const auto &in = GetInput();
  const double a = in.a;
  const double b = in.b;
  const double eps = in.eps;

  auto f = [](double x) { return ((x - 1.0) * (x - 1.0)) + 1.0; };

  std::vector<double> x_vals = {a, b};
  std::vector<double> f_vals = {f(a), f(b)};

  while ((x_vals.back() - x_vals.front()) > eps) {
    double m = 0.0;
    for (std::size_t i = 1; i < x_vals.size(); ++i) {
      m = std::max(m, std::abs(f_vals[i] - f_vals[i - 1]) / (x_vals[i] - x_vals[i - 1]));
    }

    if (m == 0.0) {
      m = 1.0;
    }

    const double r = 2.0;
    double max_r = -1e9;
    std::size_t best_idx = 1;

    for (std::size_t i = 1; i < x_vals.size(); ++i) {
      const double dx = x_vals[i] - x_vals[i - 1];
      const double df = f_vals[i] - f_vals[i - 1];
      const double r_val = (r * dx) + ((df * df) / (r * dx)) - (2.0 * (f_vals[i] + f_vals[i - 1]));

      if (r_val > max_r) {
        max_r = r_val;
        best_idx = i;
      }
    }

    const double new_x =
        (0.5 * (x_vals[best_idx] + x_vals[best_idx - 1])) - ((f_vals[best_idx] - f_vals[best_idx - 1]) / (2.0 * m));

    x_vals.insert(x_vals.begin() + static_cast<std::ptrdiff_t>(best_idx), new_x);
    f_vals.insert(f_vals.begin() + static_cast<std::ptrdiff_t>(best_idx), f(new_x));
  }

  GetOutput() = *std::ranges::min_element(f_vals);
  return true;
}

bool TelnovStronginAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace telnov_strongin_algorithm
