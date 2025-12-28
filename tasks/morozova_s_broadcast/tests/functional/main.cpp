#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "morozova_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozova_s_broadcast {

class MorozovaSBroadcastSEQFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_number = std::get<0>(params);

    switch (test_number) {
      case 1:
        input_data_ = {1, 2, 3, 4, 5};
        break;
      case 2:
        input_data_ = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        break;
      case 3:
        input_data_ = {1000, 2000, 3000, 4000, 5000};
        break;
      case 4:
        input_data_.clear();
        break;
      case 5:
        input_data_ = std::vector<int>(1000, 42);
        break;
      case 6:
        input_data_ = {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0, -1, 1};
        break;
      default:
        input_data_ = {42};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == input_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MorozovaSBroadcastSEQFuncTests, BroadcastSEQTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParamSEQ = {std::make_tuple(1, "small"),         std::make_tuple(2, "medium"),
                                               std::make_tuple(3, "large"),         std::make_tuple(4, "empty"),
                                               std::make_tuple(5, "uniform_large"), std::make_tuple(6, "edge_values")};

const auto kTestTasksSEQ =
    ppc::util::AddFuncTask<MorozovaSBroadcastSEQ, InType>(kTestParamSEQ, PPC_SETTINGS_morozova_s_broadcast);

const auto kGtestValuesSEQ = ppc::util::ExpandToValues(kTestTasksSEQ);
const auto kTestNameSEQ = MorozovaSBroadcastSEQFuncTests::PrintFuncTestName<MorozovaSBroadcastSEQFuncTests>;

INSTANTIATE_TEST_SUITE_P(BroadcastSEQTests, MorozovaSBroadcastSEQFuncTests, kGtestValuesSEQ, kTestNameSEQ);

}  // namespace

}  // namespace morozova_s_broadcast
