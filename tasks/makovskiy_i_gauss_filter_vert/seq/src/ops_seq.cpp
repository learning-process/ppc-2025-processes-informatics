#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>

namespace makovskiy_i_gauss_filter_vert {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GaussFilterSEQ::ValidationImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &[input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width * height);
}

bool GaussFilterSEQ::PreProcessingImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &[input, width, height] = GetInput();
  GetOutput().resize(width * height);
  return true;
}

bool GaussFilterSEQ::RunImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &[input, width, height] = GetInput();
  auto &output = GetOutput();

  constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  constexpr int kKernelSum = 16;

  for (int y_coord = 0; y_coord < height; ++y_coord) {
    for (int x_coord = 0; x_coord < width; ++x_coord) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          sum += GetPixel(input, x_coord + kx, y_coord + ky, width, height) *
                 kKernel[static_cast<size_t>(((ky + 1) * 3) + (kx + 1))];
        }
      }
      output[(y_coord * width) + x_coord] = sum / kKernelSum;
    }
  }
  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  return true;
}

}  // namespace makovskiy_i_gauss_filter_vert
