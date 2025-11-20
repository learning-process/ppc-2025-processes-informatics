#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"
#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  // Размер квадратной матрицы для perf-тестов
  const int kN_ = 1000;

 private:
  InType matrix_{};

  void SetUp() override {
    matrix_.assign(kN_, std::vector<int>(kN_, 1));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const OutType expected = static_cast<OutType>(kN_) * kN_;
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return matrix_;
  }
};

TEST_P(FrolovaSSumElemMatrixRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSSumElemMatrixMPI, FrolovaSSumElemMatrixSEQ>(
    PPC_SETTINGS_frolova_s_sum_elem_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaSSumElemMatrixRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FrolovaSSumElemMatrixRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace frolova_s_sum_elem_matrix
