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

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    
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

TEST_P(PylaevaSSimpleIterationMethodFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodMPI, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method),
                   ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodSEQ, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PylaevaSSimpleIterationMethodFuncTests::PrintFuncTestName<PylaevaSSimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(PylaevaSSimpleIterationMethodTests, PylaevaSSimpleIterationMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace pylaeva_s_simple_iteration_method
