#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>

namespace makovskiy_i_gauss_filter_vert {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GaussFilterSEQ::ValidationImpl() {  // NOLINT
  const auto &[input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width * height);
}

bool GaussFilterSEQ::PreProcessingImpl() {  // NOLINT
  const auto &[_, width, height] = GetInput();
  GetOutput().resize(width * height);
  return true;
}

bool GaussFilterSEQ::RunImpl() {  // NOLINT
  const auto &[input, width, height] = GetInput();
  auto &output = GetOutput();

  const std::array<int, 9> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  const int kernel_sum = 16;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int sum = 0;
      for (int k_row = -1; k_row <= 1; ++k_row) {
        for (int k_col = -1; k_col <= 1; ++k_col) {
          sum += GetPixel(input, col + k_col, row + k_row, width, height) * kernel[((k_row + 1) * 3) + (k_col + 1)];
        }
      }
      output[(row * width) + col] = sum / kernel_sum;
    }
  }
  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {  // NOLINT
  return true;
}

}  // namespace makovskiy_i_gauss_filter_vert
