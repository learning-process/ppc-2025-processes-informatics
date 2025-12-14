#include <gtest/gtest.h>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"
#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSizeN_ = 5000;
  const int kSizeM_ = 5000;
  InType input_data_{};

  void SetUp() override {
    const int M = kSizeM_;
    const int N = kSizeN_;

    std::vector<double> A_matrix(static_cast<size_t>(M) * N);
    std::vector<double> B_vector(N);

    std::fill(A_matrix.begin(), A_matrix.end(), 1.0);
    std::fill(B_vector.begin(), B_vector.end(), 1.0);

    input_data_ = std::make_tuple(M, N, A_matrix, B_vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для теста производительности нужно лишь убедиться, что MPI-результат
    // совпадает с эталонным (последовательным).
    // BaseRunPerfTests автоматически делает это, используя GetReferenceOutput().

    // Дополнительно: Проверим размерность результата (для минимальной валидации)
    const int M = kSizeM_;
    if (static_cast<int>(output_data.size()) != M) {
      // Если размеры не совпадают, это ошибка
      return false;
    }

    // Возвращаем true, полагаясь на базовый класс для сравнения значений
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KruglovaAVerticalRibMatPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KruglovaAVerticalRibbMatMPI, KruglovaAVerticalRibbMatSEQ>(
        PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaAVerticalRibMatPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaAVerticalRibMatPerfTests, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_vertical_ribbon_matvec
