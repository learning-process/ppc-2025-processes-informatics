#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"
#include "zenin_a_sum_values_by_columns_matrix/mpi/include/ops_mpi.hpp"
#include "zenin_a_sum_values_by_columns_matrix/seq/include/ops_seq.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByMatrixFunctTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_filename = params + ".txt";
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_zenin_a_sum_values_by_columns_matrix, input_filename);
    std::ifstream in_file_stream(path);
    if (!in_file_stream.is_open()) {
      throw std::runtime_error("Error while opening file: " + path);
    }

    size_t rows = 0;
    size_t columns = 0;
    in_file_stream >> rows >> columns;

    std::vector<double> matrix_data;

    double value = 0.0;
    while (in_file_stream >> value) {
      matrix_data.push_back(value);
    }

    if (matrix_data.size() != rows * columns) {
      throw std::runtime_error("Invalid matrix data");
    }

    input_data_ = std::make_tuple(rows, columns, matrix_data);
    in_file_stream.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = true;
    size_t columns = std::get<1>(input_data_);
    const std::vector<double> &matrix_data = std::get<2>(input_data_);
    size_t rows = std::get<0>(input_data_);

    if (output_data.size() != columns) {
      result = false;
      return result;
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

 private:
  InType input_data_;
};

namespace {

TEST_P(ZeninASumValuesByMatrixFunctTests, SumByColumnsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {std::string("matrix1"), std::string("matrix2"), std::string("matrix3"),
                                            std::string("matrix4"), std::string("matrix5"), std::string("matrix6"),
                                            std::string("matrix7"), std::string("matrix8"), std::string("matrix9")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZeninASumValuesByColumnsMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zenin_a_sum_values_by_columns_matrix),
                                           ppc::util::AddFuncTask<ZeninASumValuesByColumnsMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zenin_a_sum_values_by_columns_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZeninASumValuesByMatrixFunctTests::PrintFuncTestName<ZeninASumValuesByMatrixFunctTests>;

INSTANTIATE_TEST_SUITE_P(ZeninAMatrix, ZeninASumValuesByMatrixFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zenin_a_sum_values_by_columns_matrix
