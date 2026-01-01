#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "agafonov_i_torus_grid/common/include/common.hpp"
#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "agafonov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace agafonov_i_torus_grid {

class TorusGridFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    if (test_case == 1) {
      input_data_ = {123, 0, 3};
      expected_output_ = 123;
    } else if (test_case == 2) {
      input_data_ = {555, 1, 2};
      expected_output_ = 555;
    } else {
      input_data_ = {99, 0, 0};
      expected_output_ = 99;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {
TEST_P(TorusGridFuncTests, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParams = {
    std::make_tuple(1, "transfer_0_to_3"), std::make_tuple(2, "transfer_1_to_2"), std::make_tuple(3, "self_transfer")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TorusGridTaskMPI, InType>(kTestParams, PPC_SETTINGS_agafonov_i_torus_grid),
                   ppc::util::AddFuncTask<TorusGridTaskSEQ, InType>(kTestParams, PPC_SETTINGS_agafonov_i_torus_grid));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
INSTANTIATE_TEST_SUITE_P(TorusGridTests, TorusGridFuncTests, kGtestValues,
                         TorusGridFuncTests::PrintFuncTestName<TorusGridFuncTests>);
}  // namespace
}  // namespace agafonov_i_torus_grid
