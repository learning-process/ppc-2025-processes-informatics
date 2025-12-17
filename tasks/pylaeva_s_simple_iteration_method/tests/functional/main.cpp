#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

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
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(PylaevaSSimpleIterationMethodFuncTests, SimpleIterationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "size_1"),   std::make_tuple(2, "size_2"),
                                             std::make_tuple(3, "size_3"),   std::make_tuple(5, "size_5"),
                                             std::make_tuple(7, "size_7"),   std::make_tuple(10, "size_10"),
                                             std::make_tuple(15, "size_15"), std::make_tuple(20, "size_20"),
                                             std::make_tuple(30, "size_30"), std::make_tuple(50, "size_50")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method),
                                           ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    PylaevaSSimpleIterationMethodFuncTests::PrintFuncTestName<PylaevaSSimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, PylaevaSSimpleIterationMethodFuncTests, kGtestValues, kFuncTestName);

TEST(PylaevaSSimpleIterationMethodEdgeCases, InvalidInputZeroSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, InvalidInputNegativeSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, ValidInputPositiveSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, PreProcessingSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, FullExecutionSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, MinimalSizeSEQ) {
  PylaevaSSimpleIterationMethodSEQ task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, InvalidInputZeroMPI) {
  PylaevaSSimpleIterationMethodMPI task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, InvalidInputNegativeMPI) {
  PylaevaSSimpleIterationMethodMPI task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, ValidInputPositiveMPI) {
  PylaevaSSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, PreProcessingMPI) {
  PylaevaSSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, FullExecutionMPI) {
  PylaevaSSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);
}

TEST(PylaevaSSimpleIterationMethodEdgeCases, MinimalSizeMPI) {
  PylaevaSSimpleIterationMethodMPI task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);
}

}  // namespace

}  // namespace pylaeva_s_simple_iteration_method