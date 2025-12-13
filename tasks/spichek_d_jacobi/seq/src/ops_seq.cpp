#include "spichek_d_jacobi/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"

namespace spichek_d_jacobi {

SpichekDJacobiSEQ::SpichekDJacobiSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Vector{};
}

bool SpichekDJacobiSEQ::ValidationImpl() {
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    return true;
  }
  if (A[0].size() != n || b.size() != n) {
    return false;
  }
  // Простая проверка диагонального преобладания для валидации
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(A[i][i]) < 1e-12) {
      return false;
    }
  }
  return true;
}

bool SpichekDJacobiSEQ::PreProcessingImpl() {
  return true;
}

bool SpichekDJacobiSEQ::RunImpl() {
  // НЕТ вызовов MPI!

  const auto &[A, b, eps_input, max_iter_input] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    GetOutput() = Vector{};
    return true;
  }

  Vector x_k(n, 0.0);
  Vector x_k_plus_1(n, 0.0);

  int iter = 0;
  double max_diff;

  do {
    ++iter;
    max_diff = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      const auto &row = A[i];

      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += row[j] * x_k[j];
        }
      }

      x_k_plus_1[i] = (b[i] - sum) / row[i];
    }

    for (size_t i = 0; i < n; ++i) {
      max_diff = std::max(max_diff, std::abs(x_k_plus_1[i] - x_k[i]));
    }

    x_k = x_k_plus_1;

  } while (max_diff > eps_input && iter < max_iter_input);

  GetOutput() = x_k;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
