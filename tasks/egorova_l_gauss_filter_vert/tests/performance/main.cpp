#include <gtest/gtest.h>
#include <cstddef>
#include <vector>

#include "util/include/perf_test_util.hpp"

// Заголовки вашей текущей задачи
#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  // Размеры для теста производительности
  const int kRows_ = 2000;
  const int kCols_ = 2000;
  const int kChannels_ = 1;
  InType input_data_;

 public:
  void SetUp() override {
    // Инициализация структуры Image согласно common.hpp
    input_data_.rows = kRows_;
    input_data_.cols = kCols_;
    input_data_.channels = kChannels_;
    input_data_.data.assign(static_cast<size_type>(kRows_ * kCols_ * kChannels_), 128); // Тестовое заполнение
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Простейшая проверка: выходное изображение должно иметь те же размеры
    return output_data.rows == kRows_ && 
           output_data.cols == kCols_ && 
           output_data.data.size() == input_data_.data.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(EgorovaLGaussFilterVertRunPerfTests, EgorovaLGaussPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EgorovaLGaussFilterVertMPI, EgorovaLGaussFilterVertSEQ>(
        PPC_SETTINGS_egorova_l_gauss_filter_vert);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EgorovaLGaussFilterVertRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(EgorovaLGaussPerf, EgorovaLGaussFilterVertRunPerfTests, kGtestValues, kPerfTestName);

} // namespace egorova_l_gauss_filter_vert