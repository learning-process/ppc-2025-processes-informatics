#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 5'000'000;
  InType input_data_{};

  void SetUp() override {
    // глобальная строка из kCount_ символов
    GlobalData::g_data_string.clear();
    GlobalData::g_data_string.resize(kCount_);

    // заполняем её шаблоном: буквы + цифры + спецсимволы
    for (int i = 0; i < kCount_; i++) {
      if (i % 3 == 0) {
        GlobalData::g_data_string[i] = 'A';  // буква
      } else if (i % 3 == 1) {
        GlobalData::g_data_string[i] = '1';  // не буква
      } else {
        GlobalData::g_data_string[i] = '!';  // не буква
      }
    }

    // input_data_ — невнятно? понятно:
    // это просто количество букв в строке
    // правильно: kCount_ / 3
    input_data_ = kCount_ / 3;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YurkinCountingNumberPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YurkinCountingNumberPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinCountingNumberPerfTests, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
