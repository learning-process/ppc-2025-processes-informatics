#include "smyshlaev_a_gauss_filt/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_gauss_filt {

int GetPixelClamped(const InType& img, int x, int y, int ch) {
    const int w = img.width;
    const int h = img.height;
    const int c = img.channels;

    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    
    return img.data[(y * w + x) * c + ch];
}

SmyshlaevAGaussFiltSEQ::SmyshlaevAGaussFiltSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevAGaussFiltSEQ::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && GetInput().channels > 0 && !GetInput().data.empty();
}

bool SmyshlaevAGaussFiltSEQ::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAGaussFiltSEQ::RunImpl() {
  const auto &input_image = GetInput();
  auto& output_image = GetOutput();

  output_image.width = input_image.width;
  output_image.height = input_image.height;
  output_image.channels = input_image.channels;
  
  const int w = input_image.width;
  const int h = input_image.height;
  const int c = input_image.channels;
  
  output_image.data.resize(w * h * c);
  const std::vector<uint8_t> &input_data = input_image.data;


  const std::vector<int> kernel = {
      1, 2, 1,
      2, 4, 2,
      1, 2, 1
  };
  const int kernel_sum = 16;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int ch = 0; ch < c; ++ch) {
        
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int idx = x + kx;
            int idy = y + ky;

            int value = GetPixelClamped(input_image, idx, idy, ch);
            int k_value = kernel[(ky + 1) * 3 + (kx + 1)];
            sum += value * k_value;
          }
        }
        
        output_image.data[(y * w + x) * c + ch] = static_cast<uint8_t>(sum / kernel_sum);
      }
    }
  }

  return true;
}

bool SmyshlaevAGaussFiltSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_gauss_filt
