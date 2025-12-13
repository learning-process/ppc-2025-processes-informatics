#include <gtest/gtest.h>
#include <mpi.h>  // Обязательный заголовок для MPI функций

#include <algorithm>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"
#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"
#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_image_smooth {

class GaseninLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // Фиксированные параметры для тестов производительности
  const int kWidth_ = 512;
  const int kHeight_ = 512;
  const int kKernelSize_ = 5;

  InType input_data_{};

  void SetUp() override {
    // 1. ИСПРАВЛЕНИЕ: Правильная инициализация входных данных для изображения
    input_data_.width = kWidth_;
    input_data_.height = kHeight_;
    input_data_.kernel_size = kKernelSize_;
    input_data_.data.resize(kWidth_ * kHeight_);

    // Заполняем данные (например, константой 100) для предсказуемого и повторяемого прогона
    std::fill(input_data_.data.begin(), input_data_.data.end(), 100);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем данные только если они не пустые (на root-процессе для MPI, всегда для SEQ)
    if (output_data.data.empty()) {
      return true;  // Пустые данные - пропускаем проверку
    }

    // Проверяем размеры
    return output_data.width == kWidth_ && output_data.height == kHeight_ &&
           output_data.data.size() == static_cast<size_t>(kWidth_) * static_cast<size_t>(kHeight_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaseninLRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GaseninLImageSmoothMPI, GaseninLImageSmoothSEQ>(
    PPC_SETTINGS_gasenin_l_image_smooth);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaseninLRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaseninLRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_image_smooth
