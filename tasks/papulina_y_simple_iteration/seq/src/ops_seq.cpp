#include "papulina_y_simple_iteration/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"

namespace papulina_y_simple_iteration {

PapulinaYSimpleIterationSEQ::PapulinaYSimpleIterationSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>(0);
}
bool PapulinaYSimpleIterationSEQ::ValidationImpl() {
  bool basic_check = std::get<0>(GetInput()) >= 1 &&
                     DiagonalDominance(std::get<1>(GetInput()), std::get<0>(GetInput())) &&
                     DetermChecking(std::get<1>(GetInput()), std::get<0>(GetInput()));
  if (!basic_check) {
    std::cout << "Basic validation failed\n";
    return false;
  }

  size_t n = std::get<0>(GetInput());
  const auto& a_matrix = std::get<1>(GetInput());

  double norm_b = 0.0;
  for (size_t i = n; i < n; i++) {  // проверка нормы матрицы b_matrix (для сходимости к решению)
    double row_sum = 0.0;
    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        row_sum += std::abs(a_matrix[i * n + j] / a_matrix[i * n + i]);
      }
    }
    if (row_sum > norm_b) {
      norm_b = row_sum;
    }
  }
  if (norm_b >= 1.0) {
    std::cout << "WARNING: sufficient condition for convergence may not hold (norm_b = " << norm_b << " >= 1)\n";
  }
  return true;
}

bool PapulinaYSimpleIterationSEQ::PreProcessingImpl() {
  n_ = static_cast<size_t>(std::get<0>(GetInput()));
  A_.assign(n_ * n_, 0.0);
  b_.assign(n_, 0.0);
  // copying input data
  std::copy(std::get<1>(GetInput()).data(), std::get<1>(GetInput()).data() + n_ * n_, A_.data());
  std::copy(std::get<2>(GetInput()).data(), std::get<2>(GetInput()).data() + n_, b_.data());
  return true;
}
bool PapulinaYSimpleIterationSEQ::DiagonalDominance(const std::vector<double>& a_matrix, const size_t& n) {
  bool flag = true;
  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        sum += abs(a_matrix[i * n + j]);
      }
    }
    if (sum > abs(a_matrix[i * n + i])) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool PapulinaYSimpleIterationSEQ::GetDetermCheckingResult(const std::vector<double>& a_matrix, const size_t& n) {
  return DetermChecking(a_matrix, n);
}
bool PapulinaYSimpleIterationSEQ::GetDiagonalDominanceResult(const std::vector<double>& a_matrix, const size_t& n) {
  return DiagonalDominance(a_matrix, n);
}
bool PapulinaYSimpleIterationSEQ::DetermChecking(const std::vector<double>& a, const size_t& n) {
  std::vector<double> tmp = a;

  for (size_t i = 0; i < n; i++) {
    if (std::fabs(tmp[i * n + i]) < 1e-10) {  // проверка и замена нулевого диагонального элемента
      if (!FindAndSwapRow(tmp, i, n)) {
        std::cout << "Determinant is zero\n";
        return false;
      }
    }
    double pivot = tmp[i * n + i];
    for (size_t j = i; j < n; j++) {  // нормализация текущей строки
      tmp[i * n + j] /= pivot;
    }
    for (size_t j = i + 1; j < n; j++) {  // вычитание из остальных строк (зануление)
      double factor = tmp[j * n + i];
      for (size_t k = i; k < n; k++) {
        tmp[j * n + k] -= tmp[i * n + k] * factor;
      }
    }
  }

  return true;
}
bool PapulinaYSimpleIterationSEQ::FindAndSwapRow(std::vector<double>& tmp, size_t i,
                                                 size_t n) {  // вспомогательная функция для поиска и замены строки
  for (size_t j = i + 1; j < n; j++) {
    if (std::fabs(tmp[j * n + i]) > 1e-10) {
      for (size_t k = i; k < n; k++) {
        std::swap(tmp[i * n + k], tmp[j * n + k]);
      }
      return true;
    }
  }
  return false;
}
bool PapulinaYSimpleIterationSEQ::RunImpl() {
  std::vector<double> b_matrix(n_ * n_, 0.0);
  std::vector<double> d(n_, 0.0);
  // преобразование из Ax=b -> x =d+Bx
  for (size_t i = 0; i < n_; i++) {
    for (size_t j = 0; j < n_; j++) {
      if (i != j) {
        b_matrix[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
      }
    }
    d[i] = b_[i] / A_[i * n_ + i];
  }
  std::vector<double> x = d;
  for (size_t step = 0; step <= steps_count_; step++) {
    std::vector<double> x_new = ComputeNewX(b_matrix, d, x, n_);

    double sum_sq = 0.0;
    for (size_t i = 0; i < n_; i++) {
      double diff = x_new[i] - x[i];
      sum_sq += diff * diff;
    }
    double diff_norm = std::sqrt(sum_sq);

    if (diff_norm < eps_) {  // сравниваем норму разницы
      std::cout << "Result is found on the step " << step << "\n";
      GetOutput() = x_new;
      return true;
    }
    x = x_new;
  }
  std::cout << "Result isn't found\n";
  return false;
}
bool PapulinaYSimpleIterationSEQ::PostProcessingImpl() {
  return true;
}
std::vector<double> PapulinaYSimpleIterationSEQ::ComputeNewX(const std::vector<double>& b_matrix,
                                                             const std::vector<double>& d, const std::vector<double>& x,
                                                             size_t n) {
  std::vector<double> x_new(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      x_new[i] += b_matrix[i * n + j] * x[j];
    }
    x_new[i] += d[i];
  }

  return x_new;
}
}  // namespace papulina_y_simple_iteration
