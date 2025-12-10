#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"
#include <algorithm>
#include <vector>

namespace makovskiy_i_gauss_filter_vert {

GaussFilterSEQ::GaussFilterSEQ(const InType& in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GaussFilterSEQ::ValidationImpl() {
  const auto& [input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width * height);
}

bool GaussFilterSEQ::PreProcessingImpl() {
  const auto& [input, width, height] = GetInput();
  GetOutput().resize(width * height);
  return true;
}

bool GaussFilterSEQ::RunImpl() {
  const auto& [input, width, height] = GetInput();
  auto& output = GetOutput();

  const int kernel[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  const int kernel_sum = 16;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          sum += get_pixel(input, x + kx, y + ky, width, height) * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      output[y * width + x] = sum / kernel_sum;
    }
  }
  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() { return true; }

}  // namespace makovskiy_i_gauss_filter_vert