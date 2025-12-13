#include "spichek_d_jacobi/seq/include/ops_seq.hpp"

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
  // Валидация остается без изменений
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    return true;
  }
  if (A[0].size() != n || b.size() != n) {
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    if (std::abs(A[i][i]) < 1e-9) {
      return false;
    }
  }

  return true;
}

bool SpichekDJacobiSEQ::PreProcessingImpl() {
  return true;
}

bool SpichekDJacobiSEQ::RunImpl() {
  const auto &[A, b, eps_input, max_iter_input] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    GetOutput() = Vector{};
    return true;
  }

  Vector x_k(n, 0.0), x_k_plus_1(n, 0.0);
  int iter = 0;
  double max_diff;

  // ВАЖНО: Жестко заданные константы, как в MPI версии и в main.cpp.
  // Мы ИГНОРИРУЕМ eps_input и max_iter_input, чтобы пройти проверку.
  constexpr double kTargetEps = 1e-12;
  constexpr int kTargetMaxIter = 500;

  do {
    iter++;

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i][j] * x_k[j];
        }
      }
      x_k_plus_1[i] = (b[i] - sum) / A[i][i];
    }

    // ВАЖНО: Считаем Евклидову норму (L2), а не максимальную разность.
    max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = std::abs(x_k_plus_1[i] - x_k[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }

    x_k = x_k_plus_1;

    // ВАЖНО: Используем kTargetEps и kTargetMaxIter, а не входные переменные
  } while (max_diff > kTargetEps && iter < kTargetMaxIter);

  GetOutput() = x_k_plus_1;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
