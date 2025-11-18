#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

class MonteCarloFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const auto &full = info.param;

    const TestType &t = std::get<2>(full);
    std::string size_name = std::get<1>(t);

    std::string task_name = std::get<1>(full);

    return task_name + "_" + size_name;
  }

 protected:
  void SetUp() override {
    auto full_param = GetParam();
    TestType t = std::get<2>(full_param);

    int samples = std::get<0>(t);

    input_data_.a = {0.0};
    input_data_.b = {1.0};
    input_data_.samples = samples;
    input_data_.func = [](const std::vector<double> &x) { return x[0] * x[0]; };
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double expected = 1.0 / 3.0;
    return std::abs(output_data - expected) < 0.05;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MonteCarloFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kParams = {
    std::make_tuple(1000, "small"),
    std::make_tuple(5000, "medium"),
    std::make_tuple(20000, "large"),
};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationSEQ, InType>(kParams, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationMPI, InType>(kParams, PPC_SETTINGS_example_processes));

INSTANTIATE_TEST_SUITE_P(IntegrationTests, MonteCarloFuncTests, ppc::util::ExpandToValues(kTaskList),
                         MonteCarloFuncTests::PrintTestParam);

}  // namespace
}  // namespace dorofeev_i_monte_carlo_integration_processes
