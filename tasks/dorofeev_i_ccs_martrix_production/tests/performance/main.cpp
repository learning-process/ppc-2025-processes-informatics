#include <gtest/gtest.h>

#include <vector>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"
#include "dorofeev_i_ccs_martrix_production/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_ccs_martrix_production/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_ccs_martrix_production {

class CCSMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data_{};

  void SetUp() override {
    // Пятидиагональная матрица 2000x2000
    const int n = 2000;

    CCSMatrix A;
    A.rows = n;
    A.cols = n;
    A.col_ptr.resize(n + 1, 0);
    std::vector<int> row_indices;
    std::vector<double> values;

    for (int col = 0; col < n; ++col) {
      for (int offset = -2; offset <= 2; ++offset) {
        int row = col + offset;
        if (row >= 0 && row < n) {
          row_indices.push_back(row);
          values.push_back(1.0);
        }
      }
      A.col_ptr[col + 1] = row_indices.size();
    }

    A.row_indices = std::move(row_indices);
    A.values = std::move(values);

    CCSMatrix B = A;

    input_data_ = std::make_pair(A, B);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Минимальная проверка корректности
    return output_data.rows > 0 && output_data.cols > 0 &&
           output_data.col_ptr.size() == static_cast<size_t>(output_data.cols + 1);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(CCSMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevICCSMatrixProductionMPI, DorofeevICCSMatrixProductionSEQ>(
        PPC_SETTINGS_dorofeev_i_ccs_martrix_production);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = CCSMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(CCSMatrixRunModeTests, CCSMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace dorofeev_i_ccs_martrix_production
