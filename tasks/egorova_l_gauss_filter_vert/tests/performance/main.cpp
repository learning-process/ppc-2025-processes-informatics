#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <string>
#include <array>

#include "util/include/perf_test_util.hpp"

// Заголовки вашей текущей задачи
#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  // Размеры для теста производительности
  const int kRows = 2000;
  const int kCols = 2000;
  const int kChannels = 1;
  InType input_data_;

 public:
  void SetUp() override {
    // Инициализация структуры Image согласно common.hpp
    input_data_.rows = kRows;
    input_data_.cols = kCols;
    input_data_.channels = kChannels;
    input_data_.data.assign(kRows * kCols * kChannels, 128); // Тестовое заполнение
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Простейшая проверка: выходное изображение должно иметь те же размеры
    return output_data.rows == kRows && 
           output_data.cols == kCols && 
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