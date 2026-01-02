#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include <tuple>

#include "gusev_d_radix_double/common/include/common.hpp"
#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_radix_double {

class GusevDRadixDoubleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &param_info) {
    auto params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(param_info.param);
    return std::get<1>(params) + "_" + std::to_string(param_info.index);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    std::string test_name = std::get<1>(params);
    
    input_data_ = std::vector<double>(count);
    std::mt19937 gen(42);

    if (test_name.find("Positive") != std::string::npos) {
      std::uniform_real_distribution<> dis(0.1, 1000.0);
      for (int i = 0; i < count; ++i) input_data_[i] = dis(gen);
    } else if (test_name.find("Negative") != std::string::npos) {
      std::uniform_real_distribution<> dis(-1000.0, -0.1);
      for (int i = 0; i < count; ++i) input_data_[i] = dis(gen);
    } else if (test_name.find("Zero") != std::string::npos) {
      std::fill(input_data_.begin(), input_data_.end(), 0.0);
    } else {
      std::uniform_real_distribution<> dis(-1000.0, 1000.0);
      for (int i = 0; i < count; ++i) input_data_[i] = dis(gen);
    }

    if (test_name.find("Sorted") != std::string::npos && test_name.find("Reverse") == std::string::npos) {
      std::sort(input_data_.begin(), input_data_.end());
    } else if (test_name.find("Reverse") != std::string::npos) {
      std::sort(input_data_.begin(), input_data_.end(), std::greater<double>());
    }
    
    ref_output_data_ = input_data_;
    std::sort(ref_output_data_.begin(), ref_output_data_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == ref_output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_data_;
};

TEST_P(GusevDRadixDoubleFuncTests, RunTests) {
  ExecuteTest(GetParam());
}

const std::vector<TestType> kTestParams = {
    TestType{10, "SmallVector"},
    TestType{100, "MediumVector"},
    TestType{500, "LargeVector"},
    TestType{0, "EmptyVector"},
    TestType{1, "SingleElement"},
    TestType{100, "PositiveVector"},
    TestType{100, "NegativeVector"},
    TestType{100, "SortedVector"},
    TestType{100, "ReverseSortedVector"},
    TestType{50, "ZeroVector"},
    TestType{123, "OddSizeVector"}
};

const auto kTestTasksList =
    ppc::util::MakeAllFuncTasks<InType, GusevDRadixDoubleMPI, GusevDRadixDoubleSEQ>(kTestParams);

const auto kGtestValues = ppc::util::TupleToGTestValues(kTestTasksList);

const auto kPerfTestName = GusevDRadixDoubleFuncTests::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(RadixSortDoubleTests, GusevDRadixDoubleFuncTests, kGtestValues, kPerfTestName);

}  // namespace gusev_d_radix_double