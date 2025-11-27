// main(functional.cpp)
#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <string>
#include <tuple>

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"
#include "ashihmin_d_sum_of_elem/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_sum_of_elem {

class AshihminDElemVecSumFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    input_data_ = 10;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(AshihminDElemVecSumFuncTest, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AshihminDElemVecsSumMPI, InType>(kTestParam, PPC_SETTINGS_ashihmin_d_sum_of_elem),
    ppc::util::AddFuncTask<AshihminDElemVecsSumSEQ, InType>(kTestParam, PPC_SETTINGS_ashihmin_d_sum_of_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AshihminDElemVecSumFuncTest::PrintFuncTestName<AshihminDElemVecSumFuncTest>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, AshihminDElemVecSumFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ashihmin_d_sum_of_elem
