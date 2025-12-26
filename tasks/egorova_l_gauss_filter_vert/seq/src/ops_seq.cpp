#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertSEQ::EgorovaLGaussFilterVertSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.rows > 0 && in.cols > 0 && in.channels > 0 && 
         in.data.size() == static_cast<std::size_t>(in.rows * in.cols * in.channels);
}

bool EgorovaLGaussFilterVertSEQ::PreProcessingImpl() { return true; }

bool EgorovaLGaussFilterVertSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.rows = in.rows;
  out.cols = in.cols;
  out.channels = in.channels;

  std::vector<uint8_t> result_data(in.data.size());

  const std::array<std::array<float, 3>, 3> ker = {{
      {0.0625F, 0.125F, 0.0625F}, 
      {0.125F, 0.25F, 0.125F}, 
      {0.0625F, 0.125F, 0.0625F}
  }};

  for (int yy = 0; yy < in.rows; ++yy) {
    for (int xx = 0; xx < in.cols; ++xx) {
      for (int cc = 0; cc < in.channels; ++cc) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(yy + ky, 0, in.rows - 1);
            int px = std::clamp(xx + kx, 0, in.cols - 1);
            sum += static_cast<float>(in.data[((py * in.cols) + px) * in.channels + cc]) * ker[static_cast<std::size_t>(ky + 1)][static_cast<std::size_t>(kx + 1)];
          }
        }
        result_data[((yy * in.cols) + xx) * in.channels + cc] = static_cast<uint8_t>(sum);
      }
    }
  }

  out.data = std::move(result_data);
  return true;
}

bool EgorovaLGaussFilterVertSEQ::PostProcessingImpl() { return true; }

}  // namespace egorova_l_gauss_filter_vert