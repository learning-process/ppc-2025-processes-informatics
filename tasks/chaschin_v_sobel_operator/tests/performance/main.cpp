#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"
#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVRunPerfTestProcessesSO : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kCount = 1000;

  void SetUp() override {
    const int size = kCount;

    // Доступ к tuple
    auto &image = std::get<0>(input_data_);  // std::vector<std::vector<Pixel>>
    auto &width = std::get<1>(input_data_);
    auto &height = std::get<2>(input_data_);

    // Задаём размеры
    width = size;
    height = size;

    // Генерация входного изображения (детерминированные пиксели)
    image.resize(height);
    for (int i = 0; i < height; ++i) {
      image[i].resize(width);
      for (int j = 0; j < width; ++j) {
        image[i][j] = Pixel{.r = static_cast<uint8_t>((i + 1) % 256), .g = static_cast<uint8_t>((j + 2) % 256), .b = 0};
      }
    }

    // Преобразуем в градации серого
    std::vector<std::vector<float>> gray(height, std::vector<float>(width));
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        const auto &p = image[i][j];
        gray[i][j] = (0.299F * static_cast<float>(p.r)) + (0.587F * static_cast<float>(p.g)) +
                     (0.114F * static_cast<float>(p.b));
      }
    }

    // Маски Sobel
    std::vector<std::vector<int>> kKx = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    std::vector<std::vector<int>> kKy = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Подготовка expected_output_
    expected_output_.resize(height);
    for (int i = 0; i < height; ++i) {
      expected_output_[i].resize(width);
    }

    // Вычисление градиента Sobel и реконструкция Pixel
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        float gx = 0.0f;
        float gy = 0.0f;

        for (int di = -1; di <= 1; ++di) {
          int ni = i + di;
          if (ni < 0 || ni >= height) {
            continue;
          }

          for (int dj = -1; dj <= 1; ++dj) {
            int nj = j + dj;
            if (nj < 0 || nj >= width) {
              continue;
            }

            float val = gray[ni][nj];
            gx += val * static_cast<float>(kKx[di + 1][dj + 1]);
            gy += val * static_cast<float>(kKy[di + 1][dj + 1]);
          }
        }

        float grad = std::sqrt(gx * gx + gy * gy);
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0F, grad));
        expected_output_[i][j] = Pixel{.r = intensity, .g = intensity, .b = intensity};
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(ChaschinVRunPerfTestProcessesSO, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChaschinVSobelOperatorMPI, ChaschinVSobelOperatorSEQ>(
    PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChaschinVRunPerfTestProcessesSO::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChaschinVRunPerfTestProcessesSO, kGtestValues, kPerfTestName);

}  // namespace chaschin_v_sobel_operator
