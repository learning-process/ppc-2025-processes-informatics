#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_gauss_filter {

BaldinAGaussFilterSEQ::BaldinAGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAGaussFilterSEQ::ValidationImpl() {
  ImageData im = GetInput();
  return (im.width > 0 && im.height > 0 && im.channels > 0 &&
          im.pixels.size() == static_cast<size_t>(im.width * im.height * im.channels));
}

bool BaldinAGaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool BaldinAGaussFilterSEQ::RunImpl() {
  ImageData &input = GetInput();
  ImageData res = input;
  int w = input.width;
  int h = input.height;
  int c = input.channels;

  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      for (int k = 0; k < c; k++) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            int ny = std::clamp(y + dy, 0, h - 1);
            int nx = std::clamp(x + dx, 0, w - 1);

            sum += input.pixels[(ny * w + nx) * c + k] * kernel[dy + 1][dx + 1];
          }
        }
        res.pixels[(y * w + x) * c + k] = static_cast<uint8_t>(sum / 16);
      }
    }
  }

  GetOutput() = res;
  return true;
}

bool BaldinAGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_gauss_filter
