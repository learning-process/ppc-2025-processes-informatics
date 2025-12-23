#include "sannikov_i_shtrassen_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"

namespace sannikov_i_shtrassen_algorithm {

namespace {

using Matrix = std::vector<std::vector<double>>;

Matrix Add(const Matrix &a, const Matrix &b) {
  const std::size_t n = a.size();
  Matrix res(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      res[i][j] = a[i][j] + b[i][j];
    }
  }
  return res;
}

Matrix Sub(const Matrix &a, const Matrix &b) {
  const std::size_t n = a.size();
  Matrix res(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      res[i][j] = a[i][j] - b[i][j];
    }
  }
  return res;
}

Matrix MultiplyClassic(const Matrix &a, const Matrix &b) {
  const std::size_t n = a.size();
  Matrix res(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        res[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return res;
}

std::size_t NextPow2(std::size_t v) {
  std::size_t p = 1;
  while (p < v) {
    p <<= 1U;
  }
  return p;
}

Matrix PadToSize(const Matrix &src, std::size_t m) {
  const std::size_t n = src.size();
  Matrix dst(m, std::vector<double>(m, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      dst[i][j] = src[i][j];
    }
  }
  return dst;
}

Matrix CropToSize(const Matrix &src, std::size_t n) {
  Matrix dst(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      dst[i][j] = src[i][j];
    }
  }
  return dst;
}

Matrix Strassen(const Matrix &a, const Matrix &b) {
  const std::size_t n = a.size();

  // порог можно подобрать, 64 обычно ок
  if (n <= 64) {
    return MultiplyClassic(a, b);
  }

  const std::size_t k = n / 2;

  auto split = [&](const Matrix &m, std::size_t row, std::size_t col) {
    Matrix res(k, std::vector<double>(k, 0.0));
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        res[i][j] = m[i + row][j + col];
      }
    }
    return res;
  };

  const Matrix a11 = split(a, 0, 0);
  const Matrix a12 = split(a, 0, k);
  const Matrix a21 = split(a, k, 0);
  const Matrix a22 = split(a, k, k);

  const Matrix b11 = split(b, 0, 0);
  const Matrix b12 = split(b, 0, k);
  const Matrix b21 = split(b, k, 0);
  const Matrix b22 = split(b, k, k);

  const Matrix m1 = Strassen(Add(a11, a22), Add(b11, b22));
  const Matrix m2 = Strassen(Add(a21, a22), b11);
  const Matrix m3 = Strassen(a11, Sub(b12, b22));
  const Matrix m4 = Strassen(a22, Sub(b21, b11));
  const Matrix m5 = Strassen(Add(a11, a12), b22);
  const Matrix m6 = Strassen(Sub(a21, a11), Add(b11, b12));
  const Matrix m7 = Strassen(Sub(a12, a22), Add(b21, b22));

  const Matrix c11 = Add(Sub(Add(m1, m4), m5), m7);
  const Matrix c12 = Add(m3, m5);
  const Matrix c21 = Add(m2, m4);
  const Matrix c22 = Add(Sub(Add(m1, m3), m2), m6);

  Matrix res(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      res[i][j] = c11[i][j];
      res[i][j + k] = c12[i][j];
      res[i + k][j] = c21[i][j];
      res[i + k][j + k] = c22[i][j];
    }
  }

  return res;
}

}  // namespace

SannikovIShtrassenAlgorithmSEQ::SannikovIShtrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIShtrassenAlgorithmSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);

  if (a.empty() || b.empty()) {
    return false;
  }
  if (a.size() != b.size()) {
    return false;
  }
  if (a.front().empty() || b.front().empty()) {
    return false;
  }

  const std::size_t n = a.size();
  for (const auto &row : a) {
    if (row.size() != n) {
      return false;
    }
  }
  for (const auto &row : b) {
    if (row.size() != n) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SannikovIShtrassenAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &a_in = std::get<0>(input);
  const auto &b_in = std::get<1>(input);

  const std::size_t n0 = a_in.size();
  const std::size_t m = NextPow2(n0);

  const Matrix a_pad = (m == n0) ? a_in : PadToSize(a_in, m);
  const Matrix b_pad = (m == n0) ? b_in : PadToSize(b_in, m);

  const Matrix c_pad = Strassen(a_pad, b_pad);
  GetOutput() = (m == n0) ? c_pad : CropToSize(c_pad, n0);

  return !GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_shtrassen_algorithm
