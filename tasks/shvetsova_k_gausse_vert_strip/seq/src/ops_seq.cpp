#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripSEQ::ShvetsovaKGaussVertStripSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  sizeOfRib = 1;  // инициализация ширины ленты
}

bool ShvetsovaKGaussVertStripSEQ::ValidationImpl() {
  if (GetInput().first.empty()) {
    return true;
  }
  return GetInput().first.size() == GetInput().second.size();
}

bool ShvetsovaKGaussVertStripSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput().first;
  int n = static_cast<int>(matrix.size());
  if (n == 0) {
    return true;
  }

  sizeOfRib = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && std::abs(matrix[i][j]) > 1e-12) {
        int dist = std::abs(i - j);
        if (dist + 1 > sizeOfRib) {
          sizeOfRib = dist + 1;
        }
      }
    }
  }
  return true;
}

bool ShvetsovaKGaussVertStripSEQ::RunImpl() {
  const auto &A = GetInput().first;
  const auto &b_input = GetInput().second;
  int N = static_cast<int>(A.size());
  if (N == 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  // band[i] содержит элементы с i - sizeOfRib + 1 до i + sizeOfRib - 1
  std::vector<std::vector<double>> band(N);
  std::vector<int> offsets(N);
  for (int i = 0; i < N; ++i) {
    int left = std::max(0, i - sizeOfRib + 1);
    int right = std::min(N, i + sizeOfRib);
    offsets[i] = left;
    band[i].resize(right - left);
    for (int j = left; j < right; ++j) {
      band[i][j - left] = A[i][j];
    }
  }

  std::vector<double> vec = b_input;
  const double eps = std::numeric_limits<double>::epsilon() * 100.0;

  // Прямой ход с выбором главного элемента (pivot)
  for (int i = 0; i < N; ++i) {
    int offset_i = offsets[i];
    double pivot = band[i][i - offset_i];
    int pivot_row = i;
    // Находим максимум в столбце
    for (int r = i + 1; r < std::min(N, i + sizeOfRib); ++r) {
      int offset_r = offsets[r];
      double val = std::abs(band[r][i - offset_r]);
      if (val > std::abs(pivot)) {
        pivot = band[r][i - offset_r];
        pivot_row = r;
      }
    }
    if (std::abs(pivot) <= eps) {
      pivot = 1e-15;
    }

    // Меняем строки, если pivot_row != i
    if (pivot_row != i) {
      std::swap(vec[i], vec[pivot_row]);
      std::swap(band[i], band[pivot_row]);
      std::swap(offsets[i], offsets[pivot_row]);
      offset_i = offsets[i];
    }

    vec[i] /= pivot;
    for (auto &val : band[i]) {
      val /= pivot;
    }

    // Элиминация ниже
    for (int r = i + 1; r < std::min(N, i + sizeOfRib); ++r) {
      int offset_r = offsets[r];
      double factor = band[r][i - offset_r];
      if (std::abs(factor) <= eps) {
        continue;
      }
      band[r][i - offset_r] = 0.0;
      for (int j = std::max(i + 1, offset_r);
           j < std::min(offset_i + (int)band[i].size(), offset_r + (int)band[r].size()); ++j) {
        band[r][j - offset_r] -= factor * band[i][j - offset_i];
      }
      vec[r] -= factor * vec[i];
    }
  }

  // Обратный ход
  std::vector<double> x(N);
  for (int i = N - 1; i >= 0; --i) {
    x[i] = vec[i];
    int offset_i = offsets[i];
    int last_col = std::min(N, i + sizeOfRib);
    for (int j = i + 1; j < last_col; ++j) {
      x[i] -= band[i][j - offset_i] * x[j];
    }
  }

  GetOutput().assign(x.begin(), x.end());
  return true;
}

bool ShvetsovaKGaussVertStripSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace shvetsova_k_gausse_vert_strip
