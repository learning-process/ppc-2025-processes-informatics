#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "agafonov_i_torus_grid/common/include/common.hpp"
#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "agafonov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace agafonov_i_torus_grid {

class TorusGridFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  TorusGridFuncTests() = default;

  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    return std::to_string(std::get<0>(params)) + "_" + std::get<1>(params);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    if (test_case == 1) {
      input_data_ = {.value = 123, .source_rank = 0, .dest_rank = 3};
      expected_output_ = 123;
    } else if (test_case == 2) {
      input_data_ = {.value = 555, .source_rank = 1, .dest_rank = 2};
      expected_output_ = 555;
    } else {
      input_data_ = {.value = 99, .source_rank = 0, .dest_rank = 0};
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
  InType input_data_{.value = 0, .source_rank = 0, .dest_rank = 0};
  OutType expected_output_{0};
};

TEST_P(TorusGridFuncTests, RunTests) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 3> kTestParams = {
    std::make_tuple(1, "transfer_0_to_3"), std::make_tuple(2, "transfer_1_to_2"), std::make_tuple(3, "self_transfer")};

auto GetMpiValues() {
  auto tasks = ppc::util::AddFuncTask<TorusGridTaskMPI, InType>(kTestParams, PPC_SETTINGS_agafonov_i_torus_grid);
  return ppc::util::ExpandToValues(tasks);
}

auto GetSeqValues() {
  auto tasks = ppc::util::AddFuncTask<TorusGridTaskSEQ, InType>(kTestParams, PPC_SETTINGS_agafonov_i_torus_grid);
  return ppc::util::ExpandToValues(tasks);
}

INSTANTIATE_TEST_SUITE_P(MPI, TorusGridFuncTests, GetMpiValues(), TorusGridFuncTests::PrintTestParam);

INSTANTIATE_TEST_SUITE_P(SEQ, TorusGridFuncTests, GetSeqValues(), TorusGridFuncTests::PrintTestParam);

}  // namespace

}  // namespace agafonov_i_torus_grid
