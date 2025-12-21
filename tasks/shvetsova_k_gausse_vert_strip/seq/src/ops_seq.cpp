#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"

#include <cmath>
#include <utility>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripSEQ::ShvetsovaKGaussVertStripSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>(0);
}

bool ShvetsovaKGaussVertStripSEQ::ValidationImpl() {
  return GetInput().first.size() != 0;
}

bool ShvetsovaKGaussVertStripSEQ::PreProcessingImpl() {
  std::vector<double> firstRow = GetInput().first[0];
  int i = 0;
  while (firstRow[i] != 0) {
    sizeOfRib++;
    i++;
  }
  return true;
}

bool ShvetsovaKGaussVertStripSEQ::RunImpl() {
  std::vector<std::vector<double>> matrix = GetInput().first;
  std::vector<double> vec = GetInput().second;

  int sz = matrix.size();
  for (int i = 0; i < sz; ++i) {
    double lead = matrix[i][i];
    if (std::abs(lead) < 1e-12) {
      return false;
    }
    vec[i] /= lead;
    for (int j = i; j <= std::min(i + sizeOfRib - 1, sz - 1); ++j) {
      matrix[i][j] /= lead;
    }

    for (int m = i + 1; m <= std::min(i + sizeOfRib - 1, sz - 1); ++m) {
      double coef = matrix[m][i];

      vec[m] -= coef * vec[i];

      for (int s = i; s <= std::min(i + sizeOfRib - 1, sz - 1); ++s) {
        matrix[m][s] -= coef * matrix[i][s];
      }
    }
  }

  // обратный ход
  std::vector<double> result(sz);

  for (int i = sz - 1; i >= 0; --i) {
    result[i] = vec[i];
    for (int j = i + 1; j <= std::min(i + sizeOfRib - 1, sz - 1); ++j) {
      result[i] -= matrix[i][j] * result[j];
    }
  }
  GetOutput() = result;
  return true;
}

bool ShvetsovaKGaussVertStripSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_gausse_vert_strip
