#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"
#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int k_sizen_ = 6000;
  const int k_sizem_ = 6000;
  InType input_data_;

  void SetUp() override {
    const int m = k_sizem_;
    const int n = k_sizen_;

    std::vector<double> A_matrix(static_cast<size_t>(m) * n);
    std::vector<double> B_vector(n);

    std::fill(A_matrix.begin(), A_matrix.end(), 1.0);  // NOLINT(modernize-use-ranges)
    std::fill(B_vector.begin(), B_vector.end(), 1.0);  // NOLINT(modernize-use-ranges)

    input_data_ = std::make_tuple(m, n, A_matrix, B_vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int m = k_sizem_;
    return static_cast<int>(output_data.size()) == m;
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
