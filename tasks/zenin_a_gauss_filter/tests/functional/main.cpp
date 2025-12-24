#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_gauss_filter/common/include/common.hpp"
#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "zenin_a_gauss_filter/seq/include/ops_seq.hpp"

namespace zenin_a_gauss_filter {

class ZeninAGaussFilterFunctTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(full_param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(3, 3),     std::make_tuple(2, 5),   std::make_tuple(10, 70),     std::make_tuple(1, 1),
    std::make_tuple(1, 100),   std::make_tuple(100, 1), std::make_tuple(1000, 1000), std::make_tuple(10, 2),
    std::make_tuple(5, 3),     std::make_tuple(4, 5),   std::make_tuple(4, 3),       std::make_tuple(10000, 3),
    std::make_tuple(3, 10000), std::make_tuple(500, 1), std::make_tuple(1, 500)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ZeninAGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_zenin_a_gauss_filter),
                   ppc::util::AddFuncTask<ZeninAGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_zenin_a_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = ZeninAGaussFilterFunctTests::PrintFuncTestName<ZeninAGaussFilterFunctTests>;
TEST_P(ZeninAGaussFilterFunctTests, Test) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ZeninAGaussFilter, ZeninAGaussFilterFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zenin_a_gauss_filter
