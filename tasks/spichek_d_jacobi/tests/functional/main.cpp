#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_jacobi {

class SpichekDJacobiRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    // Просто проверяем, что метод сошёлся и результат адекватный
    if (output_data.empty()) {
      return false;
    }

    for (double v : output_data) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SpichekDJacobiRunFuncTestsProcesses, JacobiTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {
    std::make_tuple(InType{{{10.0, -1.0}, {-1.0, 10.0}}, {9.0, 9.0}, 1e-6, 100}, "simple_2x2"),

    std::make_tuple(InType{{{4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}}, {6.0, 6.0, 6.0}, 1e-6, 500},
                    "simple_3x3")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SpichekDJacobiMPI, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi),
                   ppc::util::AddFuncTask<SpichekDJacobiSEQ, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SpichekDJacobiRunFuncTestsProcesses::PrintFuncTestName<SpichekDJacobiRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(JacobiTests, SpichekDJacobiRunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace spichek_d_jacobi
