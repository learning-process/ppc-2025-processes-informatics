#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"
#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_elem_vec_sum {
class KurpiakovAElemVecSumFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_data_source = ppc::util::GetAbsoluteTaskPath(PPC_ID_kurpiakov_a_elem_vec_sum, param + ".txt");

    std::ifstream file(input_data_source);
    int size = 0;
    double expected = NAN;
    std::vector<double> input;
    file >> size;
    file >> expected;
    double num = 0.0;
    while (file >> num) {
      input.push_back(num);
    }
    file.close();

    input_data_ = InType(size, input);
    expected_data_ = static_cast<OutType>(expected);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (std::abs(output_data - expected_data_) <= kEps);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{0, {}};
  OutType expected_data_(0.0);
};

namespace {
TEST_P(KurpiakovAElemVecSumFuncTest, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1_empty", "test2_single", "test3_positive", "test4_zeros",
                                             "test5_mixed", "test6_border", "test7_overflow", "test8_inf",
                                             "test9_nan",   "test10_large"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KurpiakovAElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_kurpiakov_a_elem_vec_sum),
    ppc::util::AddFuncTask<KurpiakovAElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_kurpiakov_a_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KurpiakovAElemVecSumFuncTest::PrintFuncTestName<KurpiakovAElemVecSumFuncTest>;

INSTANTIATE_TEST_SUITE_P(KurpiakovAVec, KurpiakovAElemVecSumFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kurpiakov_a_elem_vec_sum
