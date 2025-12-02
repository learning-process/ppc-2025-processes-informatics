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

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "romanova_v_jacobi_method_processes/mpi/include/ops_mpi.hpp"
#include "romanova_v_jacobi_method_processes/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method_processes {

class RomanovaVJacobiMethodFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    
    //input_data_ = std::vector<double>(2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(RomanovaVJacobiMethodFuncTestsProcesses, Jacobi) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"std::make_tuple(3, "3")", "std::make_tuple(5, "5")", "std::make_tuple(7, "7")"};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RomanovaVJacobiMethodMPI, InType>(kTestParam, PPC_SETTINGS_romanova_v_jacobi_method_processes),
                   ppc::util::AddFuncTask<RomanovaVJacobiMethodSEQ, InType>(kTestParam, PPC_SETTINGS_romanova_v_jacobi_method_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RomanovaVJacobiMethodFuncTestsProcesses::PrintFuncTestName<RomanovaVJacobiMethodFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(Tests, RomanovaVJacobiMethodFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace romanova_v_jacobi_method_processes
