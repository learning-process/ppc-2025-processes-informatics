#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"
#include <vector>
#include <algorithm>

namespace gasenin_l_image_smooth {

GaseninLImageSmoothSEQ::GaseninLImageSmoothSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothSEQ::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && 
         GetInput().data.size() == static_cast<size_t>(GetInput().width * GetInput().height);
}

bool GaseninLImageSmoothSEQ::PreProcessingImpl() {
  // Инициализируем output теми же метаданными
  GetOutput() = GetInput();
  // Очищаем данные, так как будем перезаписывать
  GetOutput().data.assign(GetInput().data.size(), 0);
  return true;
}

bool GaseninLImageSmoothSEQ::RunImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();
  
  int w = in.width;
  int h = in.height;
  int k_size = in.kernel_size;
  int radius = k_size / 2;

  const uint8_t* src = in.data.data();
  uint8_t* dst = out.data.data();

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int sum = 0;
      int count = 0;

      for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
          int ny = clamp(y + ky, 0, h - 1);
          int nx = clamp(x + kx, 0, w - 1);
          
          sum += src[ny * w + nx];
          count++;
        }
      }
      dst[y * w + x] = static_cast<uint8_t>(sum / count);
    }
  }
  return true;
}

bool GaseninLImageSmoothSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth