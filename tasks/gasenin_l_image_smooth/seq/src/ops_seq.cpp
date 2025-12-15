#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"

namespace gasenin_l_image_smooth {

GaseninLImageSmoothSEQ::GaseninLImageSmoothSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothSEQ::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && GetInput().kernel_size > 0 &&
         GetInput().data.size() == static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
}

bool GaseninLImageSmoothSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool GaseninLImageSmoothSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  const int width = in.width;
  const int height = in.height;
  const int kernel_size = in.kernel_size;

  if (kernel_size <= 0) {
    return false;
  }

  const int radius = kernel_size / 2;
  const int k_sq = kernel_size * kernel_size;
  const uint8_t *src = in.data.data();
  uint8_t *dst = out.data.data();

  for (int y = 0; y < height; ++y) {
    bool is_border_y = (y < radius) || (y >= height - radius);

    for (int x = 0; x < width; ++x) {
      bool is_border_x = (x < radius) || (x >= width - radius);
      int sum = 0;

      if (!is_border_y && !is_border_x) {
        const uint8_t *row_ptr = src + (y - radius) * width + (x - radius);
        for (int ky = 0; ky < kernel_size; ++ky) {
          for (int kx = 0; kx < kernel_size; ++kx) {
            sum += row_ptr[kx];
          }
          row_ptr += width;
        }
        dst[y * width + x] = static_cast<uint8_t>(sum / k_sq);

      } else {
        int count = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
          int ny = Clamp(y + ky, 0, height - 1);
          int ny_offset = ny * width;
          for (int kx = -radius; kx <= radius; ++kx) {
            int nx = Clamp(x + kx, 0, width - 1);
            sum += src[ny_offset + nx];
            ++count;
          }
        }
        if (count > 0) {
          dst[y * width + x] = static_cast<uint8_t>(sum / count);
        } else {
          dst[y * width + x] = 0;
        }
      }
    }
  }
  return true;
}

bool GaseninLImageSmoothSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
