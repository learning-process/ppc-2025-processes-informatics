#include "krykov_e_simple_iteration/seq/include/ops_seq.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "krykov_e_simple_iteration/common/include/common.hpp"

namespace krykov_e_simple_iteration {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

KrykovESimpleIterationSEQ::KrykovESimpleIterationSEQ(const InType& in) {
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

  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[(i * n) + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[(i * n) + i];
    }

    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = x_new[i] - x[i];
      norm += diff * diff;
    }


    x = x_new;

    if (std::sqrt(norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace krykov_e_simple_iteration