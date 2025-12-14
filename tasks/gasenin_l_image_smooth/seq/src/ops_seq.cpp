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
  return GetInput().width > 0 && GetInput().height > 0 &&
         GetInput().data.size() == static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
}

bool GaseninLImageSmoothSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  GetOutput().data.assign(GetInput().data.size(), 0);
  return true;
}

bool GaseninLImageSmoothSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  const int width = in.width;
  const int height = in.height;
  const int kernel_size = in.kernel_size;
  const int radius = kernel_size / 2;

  const uint8_t *src = in.data.data();
  uint8_t *dst = out.data.data();

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      int sum = 0;
      int count = 0;

      for (int kernel_y = -radius; kernel_y <= radius; ++kernel_y) {
        for (int kernel_x = -radius; kernel_x <= radius; ++kernel_x) {
          const int ny = Clamp(row_idx + kernel_y, 0, height - 1);
          const int nx = Clamp(col_idx + kernel_x, 0, width - 1);

          sum += src[(ny * width) + nx];
          ++count;
        }
      }
      // count всегда > 0, так как kernel_size >= 3 и radius >= 1
      dst[(row_idx * width) + col_idx] = static_cast<uint8_t>(sum / count);
    }
  }
  return true;
}

bool GaseninLImageSmoothSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
