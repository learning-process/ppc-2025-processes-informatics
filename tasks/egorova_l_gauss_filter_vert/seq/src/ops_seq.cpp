#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertSEQ::EgorovaLGaussFilterVertSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.rows > 0 && input.cols > 0 && input.channels > 0 &&
         input.data.size() == static_cast<std::size_t>(input.rows) * static_cast<std::size_t>(input.cols) *
                                  static_cast<std::size_t>(input.channels);
}

bool EgorovaLGaussFilterVertSEQ::PreProcessingImpl() {
  return true;
}

bool EgorovaLGaussFilterVertSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  output.rows = input.rows;
  output.cols = input.cols;
  output.channels = input.channels;

  static constexpr std::array<float, 9> kKernel = {0.0625F, 0.125F,  0.0625F, 0.125F, 0.25F,
                                                   0.125F,  0.0625F, 0.125F,  0.0625F};

  std::vector<uint8_t> result(input.data.size());

  for (int row = 0; row < input.rows; ++row) {
    for (int col = 0; col < input.cols; ++col) {
      for (int channel = 0; channel < input.channels; ++channel) {
        double sum = 0.0;

        for (int kr = -1; kr <= 1; ++kr) {
          const int ir = std::clamp(row + kr, 0, input.rows - 1);

          for (int kc = -1; kc <= 1; ++kc) {
            const int ic = std::clamp(col + kc, 0, input.cols - 1);

            const std::size_t pixel_index =
                ((static_cast<std::size_t>(ir) * static_cast<std::size_t>(input.cols) + static_cast<std::size_t>(ic)) *
                 static_cast<std::size_t>(input.channels)) +
                static_cast<std::size_t>(channel);

            const auto kernel_index = (static_cast<std::size_t>(kr + 1) * 3U) + static_cast<std::size_t>(kc + 1);

            // Используем .at() вместо [] для проверки границ
            sum += static_cast<double>(input.data[pixel_index]) * static_cast<double>(kKernel.at(kernel_index));
          }
        }

        const std::size_t out_index =
            ((static_cast<std::size_t>(row) * static_cast<std::size_t>(input.cols) + static_cast<std::size_t>(col)) *
             static_cast<std::size_t>(input.channels)) +
            static_cast<std::size_t>(channel);

        result[out_index] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }

  output.data = std::move(result);
  return true;
}

bool EgorovaLGaussFilterVertSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_gauss_filter_vert
