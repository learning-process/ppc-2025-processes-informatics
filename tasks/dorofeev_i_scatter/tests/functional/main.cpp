#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <string>
#include <tuple>

#include "dorofeev_i_scatter/common/include/common.hpp"
#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dorofeev_i_scatter {

class DorofeevIScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    const auto &params = std::get<2>(GetParam());
    int size = std::get<0>(params);

    input_.resize(size);
    std::iota(input_.begin(), input_.end(), 0.0);  // NOLINT(modernize-use-ranges)
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    return out >= 0.0;
  }

 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<1>(param);
  }

 private:
  InType input_;
};

TEST_P(DorofeevIScatterFuncTests, ScatterCorrectness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParams = {
    std::make_tuple(4, "basic"),
};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<DorofeevIScatterMPI, InType>(kTestParams, PPC_SETTINGS_dorofeev_i_scatter),
                   ppc::util::AddFuncTask<DorofeevIScatterSEQ, InType>(kTestParams, PPC_SETTINGS_dorofeev_i_scatter));

INSTANTIATE_TEST_SUITE_P(ScatterTests, DorofeevIScatterFuncTests, ppc::util::ExpandToValues(kTasks),
                         DorofeevIScatterFuncTests::PrintFuncTestName<DorofeevIScatterFuncTests>);

}  // namespace dorofeev_i_scatter
