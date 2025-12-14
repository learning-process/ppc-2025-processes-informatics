#include "spichek_d_jacobi/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
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
  // 1. Setup
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    GetOutput() = Vector{};
    return true;
  }

  Vector x(n, 0.0);
  Vector x_new(n);

  int iter = 0;
  double max_diff = 0.0;

  // 2. Main Iteration Loop
  while (iter < max_iter) {
    ++iter;
    max_diff = 0.0;

    // 3. Calculate x_new and check convergence in one pass
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;

      // Calculate sum: S = sum(A[i][j] * x[j]) for j != i
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[i][j] * x[j];
        }
      }

      // Jacobi formula: x_i^{k+1} = (b_i - S) / A[i][i]
      x_new[i] = (b[i] - sum) / A[i][i];

      // Calculate and track the maximum difference |x_new[i] - x[i]|
      double current_diff = std::abs(x_new[i] - x[i]);

      // Replaced std::max with explicit if statement
      if (current_diff > max_diff) {
        max_diff = current_diff;
      }
    }

    // 4. Check stop condition
    if (max_diff <= eps) {
      // Convergence achieved.
      GetOutput() = x_new;
      return true;
    }

    // 5. Update: x^{k} = x^{k+1}
    x.swap(x_new);
  }

  // 6. Max iterations reached
  GetOutput() = x;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
