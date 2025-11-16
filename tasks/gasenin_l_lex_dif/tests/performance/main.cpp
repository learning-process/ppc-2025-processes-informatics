#include <gtest/gtest.h>
#include <mpi.h>  // Убедитесь, что это подключено

#include <cstdint>  // Для MPI_UINT64_T

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"
#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_lex_dif {

class GaseninLRunPerfTestsLexDif : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string long_str1;
    std::string long_str2;
    uint64_t lengths[2];

    if (rank == 0) {
      // 1. Rank 0 генерирует данные
      long_str1 = std::string(100000000, 'a');
      long_str2 = std::string(100000000, 'a');
      long_str2[5000000] = 'b';  // Различие

      lengths[0] = long_str1.length();
      lengths[1] = long_str2.length();
    }

    // 2. Rank 0 рассылает длины строк всем остальным
    MPI_Bcast(lengths, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      // 3. Ranks != 0 подготавливают (изменяют размер)
      //    свои строки для приема данных
      long_str1.resize(lengths[0]);
      long_str2.resize(lengths[1]);
    }

    // 4. Rank 0 рассылает C-string (массив char) всем остальным
    //    (Мы должны использовать &str[0], чтобы получить неконстантный указатель)
    if (lengths[0] > 0) {
      MPI_Bcast(&long_str1[0], lengths[0], MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    if (lengths[1] > 0) {
      MPI_Bcast(&long_str2[0], lengths[1], MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // 5. Теперь у ВСЕХ процессов input_data_ идентичен
    input_data_ = {long_str1, long_str2};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // str1 < str2 ('a' < 'b')
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
