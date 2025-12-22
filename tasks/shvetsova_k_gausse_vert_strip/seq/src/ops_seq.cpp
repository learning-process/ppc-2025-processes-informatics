#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripSEQ::ShvetsovaKGaussVertStripSEQ(const InType &in) : size_of_rib_(1) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
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

  size_of_rib_ = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && std::abs(matrix[i][j]) > 1e-12) {
        int dist = std::abs(i - j);
        size_of_rib_ = std::max(size_of_rib_, dist + 1);
      }
    }
  }
  return true;
}

void ShvetsovaKGaussVertStripSEQ::FindPivotAndSwap(int target_row, int n, std::vector<std::vector<double>> &band,
                                                   std::vector<int> &offsets, std::vector<double> &vec) {
  int pivot_row = target_row;
  double max_val = std::abs(band[target_row][target_row - offsets[target_row]]);

  for (int row_idx = target_row + 1; row_idx < std::min(n, target_row + size_of_rib_); ++row_idx) {
    double current_val = std::abs(band[row_idx][target_row - offsets[row_idx]]);
    if (current_val > max_val) {
      max_val = current_val;
      pivot_row = row_idx;
    }
  }

  if (pivot_row != target_row) {
    std::swap(vec[target_row], vec[pivot_row]);
    std::swap(band[target_row], band[pivot_row]);
    std::swap(offsets[target_row], offsets[pivot_row]);
  }
}

void ShvetsovaKGaussVertStripSEQ::EliminateBelow(int target_row, int n, std::vector<std::vector<double>> &band,
                                                 const std::vector<int> &offsets, std::vector<double> &vec) {
  const double eps = std::numeric_limits<double>::epsilon() * 100.0;
  int offset_i = offsets[target_row];

  for (int row_idx = target_row + 1; row_idx < std::min(n, target_row + size_of_rib_); ++row_idx) {
    int offset_r = offsets[row_idx];
    double factor = band[row_idx][target_row - offset_r];
    if (std::abs(factor) <= eps) {
      continue;
    }

    band[row_idx][target_row - offset_r] = 0.0;
    int start_j = std::max(target_row + 1, offset_r);
    int end_j = std::min(offset_i + static_cast<int>(band[target_row].size()),
                         offset_r + static_cast<int>(band[row_idx].size()));

    for (int j = start_j; j < end_j; ++j) {
      band[row_idx][j - offset_r] -= factor * band[target_row][j - offset_i];
    }
    vec[row_idx] -= factor * vec[target_row];
  }
}

bool ShvetsovaKGaussVertStripSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  int n = static_cast<int>(matrix_a.size());
  if (n == 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  std::vector<std::vector<double>> band(n);
  std::vector<int> offsets(n);
  for (int i = 0; i < n; ++i) {
    int left = std::max(0, i - size_of_rib_ + 1);
    int right = std::min(n, i + size_of_rib_);
    offsets[i] = left;
    band[i].resize(right - left);
    for (int j = left; j < right; ++j) {
      band[i][j - left] = matrix_a[i][j];
    }
  }

  std::vector<double> vec = GetInput().second;
  const double eps = std::numeric_limits<double>::epsilon() * 100.0;

  for (int i = 0; i < n; ++i) {
    FindPivotAndSwap(i, n, band, offsets, vec);
    double pivot = band[i][i - offsets[i]];
    if (std::abs(pivot) <= eps) {
      pivot = 1e-15;
    }

    vec[i] /= pivot;
    for (auto &val : band[i]) {
      val /= pivot;
    }

    EliminateBelow(i, n, band, offsets, vec);
  }

  std::vector<double> x_res(n);
  for (int i = n - 1; i >= 0; --i) {
    x_res[i] = vec[i];
    int last_col = std::min(n, i + size_of_rib_);
    for (int j = i + 1; j < last_col; ++j) {
      x_res[i] -= band[i][j - offsets[i]] * x_res[j];
    }
  }

  GetOutput().assign(x_res.begin(), x_res.end());
  return true;
}

bool ShvetsovaKGaussVertStripSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace shvetsova_k_gausse_vert_strip
