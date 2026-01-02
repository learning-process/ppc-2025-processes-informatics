#include "kiselev_i_linear_histogram_stretch/seq/include/ops_seq.hpp"

#include <cstddef>
#include <limits>
#include <vector>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"

namespace kiselev_i_linear_histogram_stretch {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  if (!in.pixels.empty()) {
    GetOutput().resize(in.pixels.size());
  }
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &img = GetInput();
  return img.width > 0 && img.height > 0 && img.pixels.size() == img.width * img.height;
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  const auto &input = GetInput().pixels;
  auto &output = GetOutput();
  uint8_t min_val = std::numeric_limits<uint8_t>::max();
  uint8_t max_val = std::numeric_limits<uint8_t>::min();

  for (uint8_t value : input) {
    if (value < min_val) {
      min_val = value;
    }
    if (value > max_val) {
      max_val = value;
    }
  }

  if (min_val == max_val) {
    for (std::size_t i = 0; i < input.size(); ++i) {
      output[i] = input[i];
    }
    return true;
  }

  const double scale = 255.0 / static_cast<double>(max_val - min_val);

  for (std::size_t i = 0; i < input.size(); ++i) {
    const double stretched = static_cast<double>(input[i] - min_val) * scale;

    output[i] = static_cast<uint8_t>(stretched + 0.5);
  }

  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_linear_histogram_stretch
