#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"
#include "eremin_v_strongin_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVRunFuncTestsStronginAlgorithm : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string result = "test_" + std::to_string(std::get<0>(test_param)) + "_from_" +
                         std::to_string(std::get<1>(test_param)) + "_to_" + std::to_string(std::get<2>(test_param)) +
                         "_steps_" + std::to_string(std::get<3>(test_param));
    std::ranges::replace(result, '.', '_');
    std::ranges::replace(result, '-', 'm');
    return result;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    double lower_bound = std::get<1>(params);
    double upper_bound = std::get<2>(params);
    int steps = std::get<3>(params);
    auto function = std::get<4>(params);

    expected_result_ = std::get<5>(params);

    input_data_ = std::make_tuple(lower_bound, upper_bound, steps, function);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    double lower_bound = std::get<1>(params);
    double upper_bound = std::get<2>(params);
    int steps = std::get<3>(params);
    auto function = std::get<4>(params);
    std::cout << "Check: lower_bound: " << lower_bound << ", upper_bound: " << upper_bound << ", steps: " << steps
              << "\n";
    double tolerance = std::max(std::abs(expected_result_) * 0.01, 1e-8);
    std::cout << "Check: output_data: " << output_data << ", expected_result_: " << expected_result_ << "\n\n\n";
    return std::abs(output_data - expected_result_) <= tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_{};
};

namespace {

TEST_P(EreminVRunFuncTestsStronginAlgorithm, StronginAlgorithm) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(1, -5.0, 5.0, 1000, [](double x) { return x * x; }, 0.0),

    std::make_tuple(2, 2.0, 14.0, 5000, [](double x) { return (x - 2) * (x - 2); }, 2.0),

    std::make_tuple(3, 0.0, 8.0, 11000, [](double x) { return std::sin(x); }, 1.5 * M_PI),

    std::make_tuple(4, -30.0, 30.0, 2000, [](double x) { return x * x * x * x - 3 * x * x; }, std::sqrt(1.5)),

    std::make_tuple(5, -2.0, 100.0, 3000, [](double x) { return std::exp(x); }, -2.0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EreminVStronginAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_eremin_v_strongin_algorithm),
    ppc::util::AddFuncTask<EreminVStronginAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_eremin_v_strongin_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    EreminVRunFuncTestsStronginAlgorithm::PrintFuncTestName<EreminVRunFuncTestsStronginAlgorithm>;

INSTANTIATE_TEST_SUITE_P(StronginAlgorithmTests, EreminVRunFuncTestsStronginAlgorithm, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace eremin_v_strongin_algorithm
