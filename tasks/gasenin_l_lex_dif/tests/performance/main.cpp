#include <gtest/gtest.h>

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"
#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_lex_dif {

class GaseninLRunPerfTestsLexDif : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    // Генерируем длинные строки для тестирования производительности
    std::string long_str1(1000000000, 'a');
    std::string long_str2(1000000000, 'a');
    long_str2[9999] = 'b';  // Различие в последнем символе

    input_data_ = {long_str1, long_str2};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // str1 < str2 (последний символ 'a' < 'b')
    // Ожидаем результат -1
    return output_data == -1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaseninLRunPerfTestsLexDif, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaseninLLexDifMPI, GaseninLLexDifSEQ>(PPC_SETTINGS_gasenin_l_lex_dif);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaseninLRunPerfTestsLexDif::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LexDifPerfTests, GaseninLRunPerfTestsLexDif, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_lex_dif
