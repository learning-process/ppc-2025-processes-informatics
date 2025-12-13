#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "krykov_e_simple_iteration/common/include/common.hpp"
#include "krykov_e_simple_iteration/mpi/include/ops_mpi.hpp"
#include "krykov_e_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_simple_iteration {

class KrykovESimpleIterationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KrykovESimpleIterationFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    (void)test_param;
    static int counter = 1;
    return "SLAE_Test_" + std::to_string(counter++);
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_output_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double eps = 1e-5;
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > eps) {
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
  OutType expected_output_{};
};

namespace {

// 1) Простая 1x1 система
// x = 1
const TestType test_1 = {{1, {1.0}, {1.0}}, {1.0}};

// 2) Диагональная система 2x2
// 2x = 4
// 3y = 6
const TestType test_2 = {{2, {2.0, 0.0, 0.0, 3.0}, {4.0, 6.0}}, {2.0, 2.0}};

// 3) Простая 2x2 система
// 4x + y = 9
// x + 3y = 5
const TestType test_3 = {{2, {4.0, 1.0, 1.0, 3.0}, {9.0, 5.0}}, {2.0, 1.0}};

// 4) 3x3 диагонально доминирующая система
const TestType test_4 = {{3, {10.0, 1.0, 1.0, 2.0, 10.0, 1.0, 2.0, 2.0, 10.0}, {12.0, 13.0, 14.0}}, {1.0, 1.0, 1.0}};

TEST_P(KrykovESimpleIterationFuncTests, SimpleIterationTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {test_1, test_2, test_3, test_4};
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KrykovESimpleIterationMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_simple_iteration),
    ppc::util::AddFuncTask<KrykovESimpleIterationSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_simple_iteration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovESimpleIterationFuncTests::PrintFuncTestName<KrykovESimpleIterationFuncTests>;

INSTANTIATE_TEST_SUITE_P(SimpleIterationTests, KrykovESimpleIterationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krykov_e_simple_iteration
