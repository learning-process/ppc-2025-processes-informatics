#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"
#include "zenin_a_sum_values_by_columns_matrix/mpi/include/ops_mpi.hpp"
#include "zenin_a_sum_values_by_columns_matrix/seq/include/ops_seq.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string input_data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_zenin_a_sum_values_by_columns_matrix, "mat_perf.txt");

    std::ifstream file(input_data_source);
    if (!file.is_open()) {
      throw std::runtime_error("Error while opening file: " + input_data_source);
    }
    size_t rows = 0;
    size_t columns = 0;
    std::vector<double> input;
    file >> rows;
    file >> columns;
    double value = 0.0;
    while (file >> value) {
      input.push_back(value);
    }
    if (input.size() != rows * columns) {
      throw std::runtime_error("Invalid matrix data");
    }
    input_data_ = std::make_tuple(rows, columns, input);
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool res = true;
    size_t columns = std::get<1>(input_data_);
    const std::vector<double> &matrix_data = std::get<2>(input_data_);
    size_t rows = std::get<0>(input_data_);
    if (output_data.size() != columns) {
      res = false;
      return res;
    }
    std::vector<double> expected_sums(columns, 0.0);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < columns; ++column) {
        expected_sums[column] += matrix_data[(row * columns) + column];
      }
    }
    for (size_t column = 0; column < columns; ++column) {
      if (std::abs(output_data[column] - expected_sums[column]) > 1e-12) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZeninASumValuesByMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZeninASumValuesByColumnsMatrixMPI, ZeninASumValuesByColumnsMatrixSEQ>(
        PPC_SETTINGS_zenin_a_sum_values_by_columns_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZeninASumValuesByMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ZeninAPerfTestMatrix, ZeninASumValuesByMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace zenin_a_sum_values_by_columns_matrix
