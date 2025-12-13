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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // В SEQ версии вычисления делает только 0-й ранг.
  // Остальные сразу выходят, чтобы не мешать замерам времени
  // (иначе фреймворк может ждать их или суммировать время некорректно).
  if (rank != 0) {
    GetOutput() = Vector{};
    return true;
  }

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

  // ИСПОЛЬЗУЕМ ВХОДНЫЕ ПАРАМЕТРЫ (мы убрали хардкод)
  double target_eps = eps_input;
  int target_max_iter = max_iter_input;

  do {
    iter++;
    max_diff = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      double a_ii = A[i][i];
      // Оптимизация: выносим указатель на строку
      const auto &row_i = A[i];

      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          sum += row_i[j] * x_k[j];
        }
      }
      x_k_plus_1[i] = (b[i] - sum) / a_ii;

      // Считаем норму разности сразу здесь
      double diff = std::abs(x_k_plus_1[i] - x_k[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }

    x_k = x_k_plus_1;

  } while (max_diff > target_eps && iter < target_max_iter);

  GetOutput() = x_k;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
