#include "krykov_e_simple_iteration/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <ranges>
#include <string>
#include <cmath>

#include "krykov_e_simple_iteration/common/include/common.hpp"

namespace krykov_e_simple_iteration {
KrykovESimpleIterationSEQ::KrykovESimpleIterationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationSEQ::ValidationImpl() {
  const auto& [n, A, b] = GetInput();
  return n > 0 && A.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationSEQ::RunImpl() {
  const auto& [n, A, b] = GetInput();

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  constexpr double eps = 1e-5;
  constexpr int max_iter = 100;

  for (int iter = 0; iter < max_iter; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i * n + i];
    }

    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      diff = std::max(diff, std::abs(x_new[i] - x[i]));
    }

    x = x_new;
    if (diff < eps) break;
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace krykov_e_simple_iteration