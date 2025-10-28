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
    return std::to_string(std::get<0>(test_param)) + "____" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    input_data_ = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    output_data_ = {1, 4, 7};
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

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

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
