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

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "kichanova_k_count_letters_in_str/mpi/include/ops_mpi.hpp"
#include "kichanova_k_count_letters_in_str/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_count_letters_in_str {

class KichanovaKCountLettersInStrFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string input_str = std::get<1>(test_param);
    std::string safe_name;

    for (char c : input_str) {
      if (std::isalnum(static_cast<unsigned char>(c))) {
        safe_name += c;
      } else {
        safe_name += '_';
      }
    }

    return std::to_string(std::get<0>(test_param)) + "_" + safe_name;
  }

protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(
        ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_string_ = std::get<1>(params);
    expected_output_ = CalculateExpectedCount(input_string_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final { return input_string_; }

private:
  int CalculateExpectedCount(const std::string &str) {
    int count = 0;
    for (char c : str) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        count++;
      }
    }
    return count;
  }

  std::string input_string_;
  int expected_output_;
};

namespace {

TEST_P(KichanovaKCountLettersInStrFuncTests, CountLettersInString) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(5, "Hello"),
    std::make_tuple(10, "Hello World"),
    std::make_tuple(8, "Test123!"),
    std::make_tuple(0, "123!@#"),
    std::make_tuple(0, " "),
    std::make_tuple(4, "a.b,c!d"),
    std::make_tuple(50, std::string(100, 'a') + std::string(100, '1')),
    std::make_tuple(1000, std::string(1000, 'x')),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KichanovaKCountLettersInStrMPI, InType>(
        kTestParam, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<KichanovaKCountLettersInStrSEQ, InType>(
        kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KichanovaKCountLettersInStrFuncTests::PrintFuncTestName<
        KichanovaKCountLettersInStrFuncTests>;

INSTANTIATE_TEST_SUITE_P(StringTests, KichanovaKCountLettersInStrFuncTests,
                         kGtestValues, kPerfTestName);

} // namespace

} // namespace kichanova_k_count_letters_in_str
