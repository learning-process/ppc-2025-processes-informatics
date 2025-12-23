#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "krykov_e_multistep_sad/common/include/common.hpp"
#include "krykov_e_multistep_sad/mpi/include/ops_mpi.hpp"
#include "krykov_e_multistep_sad/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_multistep_sad {

class KrykovEMultistepSADFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KrykovEMultistepSADFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    (void)test_param;
    static int counter = 1;
    return "GlobalOpt2D_Test_" + std::to_string(counter++);
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_output_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kEps = 1e-4;

    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

// f(x,y) = x^2 + y^2
const TestType kTest1 = {{[](double x, double y) { return x * x + y * y; }, -1.0, 1.0, -1.0, 1.0}, {0.0, 0.0, 0.0}};

// f(x,y) = (x-1)^2 + (y+2)^2
const TestType kTest2 = {
    {[](double x, double y) { return (x - 1.0) * (x - 1.0) + (y + 2.0) * (y + 2.0); }, -5.0, 5.0, -5.0, 5.0},
    {1.0, -2.0, 0.0}};

// f(x,y) = (x+1)^2 + (y-3)^2
const TestType kTest3 = {
    {[](double x, double y) { return (x + 1.0) * (x + 1.0) + (y - 3.0) * (y - 3.0); }, -5.0, 5.0, -5.0, 5.0},
    {-1.0, 3.0, 0.0}};

TEST_P(KrykovEMultistepSADFuncTests, GlobalOptimizationTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {kTest1, kTest2, kTest3};
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KrykovEMultistepSADMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_multistep_sad),
    ppc::util::AddFuncTask<KrykovEMultistepSADSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_multistep_sad));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovEMultistepSADFuncTests::PrintFuncTestName<KrykovEMultistepSADFuncTests>;

INSTANTIATE_TEST_SUITE_P(GlobalOptimizationTests, KrykovEMultistepSADFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krykov_e_multistep_sad
