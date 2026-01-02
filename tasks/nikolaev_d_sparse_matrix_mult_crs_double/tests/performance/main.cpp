#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

#include "nikolaev_d_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "nikolaev_d_sparse_matrix_mult_crs_double/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_sparse_matrix_mult_crs_double/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikolaev_d_sparse_matrix_mult_crs_double {

class NikolaevDSparseMatrixMultCrsDoublePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    static size_t test_index = 0;
    static constexpr std::array<InType, 4> kTestSizes = {27, 28, 29, 30};
    InType result = kTestSizes.at(test_index % kTestSizes.size());
    test_index++;
    return result;
  }
};

TEST_P(NikolaevDSparseMatrixMultCrsDoublePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikolaevDSparseMatrixMultCrsDoubleMPI, NikolaevDSparseMatrixMultCrsDoubleSEQ>(
        PPC_SETTINGS_nikolaev_d_sparse_matrix_mult_crs_double);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikolaevDSparseMatrixMultCrsDoublePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikolaevDSparseMatrixMultCrsDoublePerfTests, kGtestValues, kPerfTestName);

}  // namespace nikolaev_d_sparse_matrix_mult_crs_double
