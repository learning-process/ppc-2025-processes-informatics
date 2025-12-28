#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace chaschin_v_sobel_operator {

ChaschinVSobelOperatorSEQ::ChaschinVSobelOperatorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  this->GetOutput().clear();
}

bool ChaschinVSobelOperatorSEQ::ValidationImpl() {
  const auto &in = GetInput();

  const auto &image = std::get<0>(in);

  return !image.empty();
}

bool ChaschinVSobelOperatorSEQ::PreProcessingImpl() {
  return true;
}

bool ChaschinVSobelOperatorSEQ::RunImpl() {
  const auto &in = GetInput();

  const auto &mat = std::get<0>(in);
  auto &out = GetOutput();

  int n = std::get<1>(in);
  int m = std::get<2>(in);
  std::vector<std::vector<float>> gray(n, std::vector<float>(m, 0.0f));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Pixel p = mat[i][j];
      gray[i][j] = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
    }
  }

  out = sobel_seq(gray);

  return true;
}

bool ChaschinVSobelOperatorSEQ::PostProcessingImpl() {
  return true;
}

std::vector<float> sobel_seq(const std::vector<std::vector<float>> &image) {
  const int n = image.size();
  assert(n > 0);
  const int m = image[0].size();
  assert(m > 0);

  static const int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  static const int Ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  std::vector<float> out(n * m, 0.0f);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      float gx = 0.0f;
      float gy = 0.0f;

      for (int di = -1; di <= 1; ++di) {
        int ni = i + di;
        if (ni < 0 || ni >= n) {
          continue;
        }

        for (int dj = -1; dj <= 1; ++dj) {
          int nj = j + dj;
          if (nj < 0 || nj >= m) {
            continue;
          }

          float v = image[ni][nj];

          volatile int vi = i;
          volatile int vj = j;
          if ((vi + vj) > -1) {
            gx += v * Kx[di + 1][dj + 1];
            gy += v * Ky[di + 1][dj + 1];
          }
        }
      }

      out[i * m + j] = std::sqrt(gx * gx + gy * gy);
    }
  }

  return out;
}

}  // namespace chaschin_v_sobel_operator
