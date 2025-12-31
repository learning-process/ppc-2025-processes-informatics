#include "egashin_k_iterative_simple/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"

namespace egashin_k_iterative_simple {

EgashinKIterativeSimpleSEQ::EgashinKIterativeSimpleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input = GetInput();
  input.A.clear();
  input.A.reserve(in.A.size());
  for (const auto &row : in.A) {
    input.A.push_back(row);
  }
  input.b.clear();
  input.b.reserve(in.b.size());
  input.b.insert(input.b.end(), in.b.begin(), in.b.end());
  input.x0.clear();
  input.x0.reserve(in.x0.size());
  input.x0.insert(input.x0.end(), in.x0.begin(), in.x0.end());
  input.tolerance = in.tolerance;
  input.max_iterations = in.max_iterations;
  GetOutput() = std::vector<double>(in.A.size(), 0.0);
}

bool EgashinKIterativeSimpleSEQ::ValidationImpl() {
  const auto &input = GetInput();
  std::size_t n = input.A.size();

  if (n == 0) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (input.A[i].size() != n) {
      return false;
    }
  }

  if (input.b.size() != n || input.x0.size() != n) {
    return false;
  }

  if (input.tolerance <= 0 || input.max_iterations <= 0) {
    return false;
  }

  return true;
}

bool EgashinKIterativeSimpleSEQ::PreProcessingImpl() {
  return true;
}

double EgashinKIterativeSimpleSEQ::CalculateTau(const std::vector<std::vector<double>> &matrix) {
  double max_row_sum = 0.0;
  std::size_t n = matrix.size();

  for (std::size_t i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      row_sum += std::abs(matrix[i][j]);
    }
    max_row_sum = std::max(max_row_sum, row_sum);
  }

  if (max_row_sum < 1e-10) {
    return 0.1;
  }
  return 1.0 / (max_row_sum + 1.0);
}

double EgashinKIterativeSimpleSEQ::CalculateNorm(const std::vector<double> &v) {
  double norm = 0.0;
  for (double val : v) {
    norm += val * val;
  }
  return std::sqrt(norm);
}

bool EgashinKIterativeSimpleSEQ::CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new,
                                                  double tol) {
  double diff_norm = 0.0;
  for (std::size_t i = 0; i < x_old.size(); ++i) {
    double diff = x_new[i] - x_old[i];
    diff_norm += diff * diff;
  }
  return std::sqrt(diff_norm) < tol;
}

bool EgashinKIterativeSimpleSEQ::RunImpl() {
  const auto &input = GetInput();
  std::size_t n = input.A.size();

  double tau = CalculateTau(input.A);

  std::vector<double> x = input.x0;
  std::vector<double> x_new(n);

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    for (std::size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        ax_i += input.A[i][j] * x[j];
      }
      x_new[i] = x[i] + (tau * (input.b[i] - ax_i));
    }

    if (CheckConvergence(x, x_new, input.tolerance)) {
      GetOutput() = x_new;
      return true;
    }

    x = x_new;
  }

  GetOutput() = x_new;
  return true;
}

bool EgashinKIterativeSimpleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egashin_k_iterative_simple
