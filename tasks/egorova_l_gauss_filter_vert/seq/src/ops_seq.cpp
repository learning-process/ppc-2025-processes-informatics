#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertSEQ::EgorovaLGaussFilterVertSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.rows > 0 && in.cols > 0 && in.channels > 0 &&
         in.data.size() == (static_cast<std::size_t>(in.rows) * in.cols * in.channels);
}

bool EgorovaLGaussFilterVertSEQ::PreProcessingImpl() {
  return true;
}

bool EgorovaLGaussFilterVertSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.rows = in.rows;
  out.cols = in.cols;
  out.channels = in.channels;

  std::vector<uint8_t> result_data(in.data.size());

  const std::array<float, 9> ker = {0.0625F, 0.125F, 0.0625F, 0.125F, 0.25F, 0.125F, 0.0625F, 0.125F, 0.0625F};
  const float *ker_ptr = ker.data();

  for (int yy = 0; yy < in.rows; ++yy) {
    for (int xx = 0; xx < in.cols; ++xx) {
      for (int cc = 0; cc < in.channels; ++cc) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(yy + ky, 0, in.rows - 1);
            int px = std::clamp(xx + kx, 0, in.cols - 1);

            std::size_t in_idx =
                ((((static_cast<std::size_t>(py) * static_cast<std::size_t>(in.cols)) + static_cast<std::size_t>(px)) *
                  static_cast<std::size_t>(in.channels)) +
                 static_cast<std::size_t>(cc));

            std::size_t k_idx = (static_cast<std::size_t>(ky + 1) * 3) + static_cast<std::size_t>(kx + 1);

            // Использование указателя на данные массива убирает предупреждение о неконстантном индексе
            sum += (static_cast<float>(in.data[in_idx]) * ker_ptr[k_idx]);
          }
        }
        std::size_t out_idx =
            ((((static_cast<std::size_t>(yy) * static_cast<std::size_t>(in.cols)) + static_cast<std::size_t>(xx)) *
              static_cast<std::size_t>(in.channels)) +
             static_cast<std::size_t>(cc));

        result_data[out_idx] = static_cast<uint8_t>(sum);
      }
    }
  }

  out.data = std::move(result_data);
  return true;
}

bool EgorovaLGaussFilterVertSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_gauss_filter_vert
