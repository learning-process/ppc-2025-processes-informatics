#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"
#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVRunPerfTestProcessesSO : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_count = 5000;
  InType input_data;

  void SetUp() override {
    // Создаем квадратное изображение
    input_data_.resize(k_count);
    for (int i = 0; i < k_count; ++i) {
      input_data_[i].resize(k_count);
      for (int j = 0; j < k_count; ++j) {
        // Детерминированное значение: только красный канал
        input_data_[i][j] = Pixel{static_cast<uint8_t>((i + 1) % 256), static_cast<uint8_t>((j + 2) % 256), 0};
      }
    }

    // Маски Sobel
    static const int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int Ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Вычисляем Sobel (градиент) для каждого пикселя
    expected_output_.resize(k_count * k_count);
    for (int i = 0; i < k_count; ++i) {
      for (int j = 0; j < k_count; ++j) {
        float gx = 0.0f, gy = 0.0f;
        for (int di = -1; di <= 1; ++di) {
          int ni = i + di;
          if (ni < 0 || ni >= k_count) {
            continue;
          }
          for (int dj = -1; dj <= 1; ++dj) {
            int nj = j + dj;
            if (nj < 0 || nj >= k_count) {
              continue;
            }
            float val = static_cast<float>(input_data_[ni][nj].r);
            gx += val * Kx[di + 1][dj + 1];
            gy += val * Ky[di + 1][dj + 1];
          }
        }
        expected_output_[i * k_count + j] = std::sqrt(gx * gx + gy * gy);
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

  TEST_P(ChaschinVRunPerfTestProcessesSO, RunPerfModesSO) {
    ExecuteTest(GetParam());
  }

  const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChaschinVSobelOperatorMPI, ChaschinVSobelOperatorSEQ>(
      PPC_SETTINGS_example_processes);

  const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

  const auto kPerfTestName = ChaschinVRunPerfTestProcessesSO::CustomPerfTestName;

  INSTANTIATE_TEST_SUITE_P(RunModeTests, ChaschinVRunPerfTestProcessesSO, kGtestValues, kPerfTestName);

}  // namespace chaschin_v_sobel_operator
