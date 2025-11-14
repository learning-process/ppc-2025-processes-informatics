#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "____" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t ROWS = std::get<0>(param);
    size_t COLS = std::get<1>(param);
    input_data_ = std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ROWS, COLS);
    if (ROWS == 4 && COLS == 3) {
      output_data_ = {1, 4, 7, 10};
    }
    if (ROWS == 3 && COLS == 4) {
      input_data_ = std::make_tuple(std::vector<int>{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, ROWS, COLS);
      output_data_ = {9, 5, 1};
    }
    if (ROWS == 6 && COLS == 2) {
      output_data_ = {1, 3, 5, 7, 9, 11};
    }
    if (ROWS == 12 && COLS == 1) {
      output_data_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    }
    if (ROWS == 1 && COLS == 12) {
      output_data_ = {1};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

namespace {

TEST_P(LevonychevIMinValRowsMatrixFuncTests, MinValRowsMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(4, 3), std::make_tuple(3, 4), std::make_tuple(6, 2),
                                            std::make_tuple(12, 1), std::make_tuple(1, 12)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LevonychevIMinValRowsMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_min_val_rows_matrix),
                                           ppc::util::AddFuncTask<LevonychevIMinValRowsMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_min_val_rows_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LevonychevIMinValRowsMatrixFuncTests::PrintFuncTestName<LevonychevIMinValRowsMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValRowsMatrixTests, LevonychevIMinValRowsMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace levonychev_i_min_val_rows_matrix
