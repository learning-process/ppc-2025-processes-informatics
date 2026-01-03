#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace goriacheva_k_strassen_algorithm {

using Matrix = std::vector<std::vector<double>>;

struct InType {
  Matrix A;
  Matrix B;
};

using OutType = Matrix;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline bool is_square(const Matrix& M) {
  if (M.empty()) return false;
  std::size_t n = M.size();
  for (const auto& r : M)
    if (r.size() != n) return false;
  return true;
}

inline std::size_t next_power_of_two(std::size_t n) {
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

inline Matrix pad_matrix(const Matrix& A, std::size_t new_size) {
  Matrix R(new_size, std::vector<double>(new_size, 0.0));
  for (std::size_t i = 0; i < A.size(); ++i)
    for (std::size_t j = 0; j < A.size(); ++j)
      R[i][j] = A[i][j];
  return R;
}

inline Matrix crop_matrix(const Matrix& A, std::size_t size) {
  Matrix R(size, std::vector<double>(size));
  for (std::size_t i = 0; i < size; ++i)
    for (std::size_t j = 0; j < size; ++j)
      R[i][j] = A[i][j];
  return R;
}

inline Matrix add(const Matrix& A, const Matrix& B) {
  std::size_t n = A.size();
  Matrix C(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      C[i][j] = A[i][j] + B[i][j];
  return C;
}

inline Matrix sub(const Matrix& A, const Matrix& B) {
  std::size_t n = A.size();
  Matrix C(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      C[i][j] = A[i][j] - B[i][j];
  return C;
}

inline Matrix naive_multiply(const Matrix& A, const Matrix& B) {
  std::size_t n = A.size();
  Matrix C(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

inline Matrix strassen(const Matrix& A, const Matrix& B,
                       std::size_t threshold = 128) {
  std::size_t n = A.size();
  if (n <= threshold)
    return naive_multiply(A, B);

  std::size_t k = n / 2;

  Matrix A11(k, std::vector<double>(k)), A12(k, std::vector<double>(k));
  Matrix A21(k, std::vector<double>(k)), A22(k, std::vector<double>(k));
  Matrix B11(k, std::vector<double>(k)), B12(k, std::vector<double>(k));
  Matrix B21(k, std::vector<double>(k)), B22(k, std::vector<double>(k));

  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + k];
      A21[i][j] = A[i + k][j];
      A22[i][j] = A[i + k][j + k];
      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + k];
      B21[i][j] = B[i + k][j];
      B22[i][j] = B[i + k][j + k];
    }

  Matrix M1 = strassen(add(A11, A22), add(B11, B22), threshold);
  Matrix M2 = strassen(add(A21, A22), B11, threshold);
  Matrix M3 = strassen(A11, sub(B12, B22), threshold);
  Matrix M4 = strassen(A22, sub(B21, B11), threshold);
  Matrix M5 = strassen(add(A11, A12), B22, threshold);
  Matrix M6 = strassen(sub(A21, A11), add(B11, B12), threshold);
  Matrix M7 = strassen(sub(A12, A22), add(B21, B22), threshold);

  Matrix C(n, std::vector<double>(n));
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) {
      C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
      C[i][j + k]     = M3[i][j] + M5[i][j];
      C[i + k][j]     = M2[i][j] + M4[i][j];
      C[i + k][j + k] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
    }

  return C;
}

}  // namespace goriacheva_k_strassen_algorithm
