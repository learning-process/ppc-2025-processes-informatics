#include <gtest/gtest.h>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_jacobi {

class JacobiPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_;

  void SetUp() override {
    size_t n = 1000;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<double> b(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
      A[i][i] = 2.0;
    }

    input_ = {A, b, 1e-6, 1000};
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    return !out.empty();
  }
};

TEST_P(JacobiPerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    JacobiPerf, JacobiPerfTests,
    ppc::util::TupleToGTestValues(
        ppc::util::MakeAllPerfTasks<InType, SpichekDJacobiMPI, SpichekDJacobiSEQ>(PPC_SETTINGS_spichek_d_jacobi)));

}  // namespace spichek_d_jacobi
