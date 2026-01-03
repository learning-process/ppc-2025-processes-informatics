#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "dorofeev_i_ccs_matrix_production/common/include/common.hpp"
#include "dorofeev_i_ccs_matrix_production/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_ccs_matrix_production/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_ccs_matrix_production {

class CCSMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;

  void SetUp() override {
    // Пятидиагональная матрица 2000x2000
    const int n = 2000;

    CCSMatrix a;
    a.rows = n;
    a.cols = n;
    a.col_ptr.resize(n + 1, 0);
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
      a.col_ptr[col + 1] = static_cast<int>(row_indices.size());
    }

    a.row_indices = std::move(row_indices);
    a.values = std::move(values);

    CCSMatrix b = a;

    input_data = std::make_pair(a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Минимальная проверка корректности
    return output_data.rows > 0 && output_data.cols > 0 &&
           static_cast<int>(output_data.col_ptr.size()) == output_data.cols + 1;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(CCSMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevICCSMatrixProductionMPI, DorofeevICCSMatrixProductionSEQ>(
        PPC_SETTINGS_dorofeev_i_ccs_matrix_production);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = CCSMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(CCSMatrixRunModeTests, CCSMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace dorofeev_i_ccs_matrix_production
