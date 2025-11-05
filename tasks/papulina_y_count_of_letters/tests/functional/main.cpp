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

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"
#include "papulina_y_count_of_letters/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace papulina_y_count_of_letters {

class PapulinaYRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::string(std::get<0>(std::get<0>(params)));
    expectedResult_ = std::get<1>(std::get<0>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expectedResult_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = "";
  OutType expectedResult_ = 0;
};

namespace {

TEST_P(PapulinaYRunFuncTestsProcesses, CountOfLetters) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(std::make_tuple("", 0), "test1"),
    std::make_tuple(std::make_tuple("abcd", 4), "test2"),
    std::make_tuple(std::make_tuple("aabcd123abcd123abcd", 13), "test3"),
    std::make_tuple(std::make_tuple("abcd_____________123abcd", 8), "test4"),
    std::make_tuple(std::make_tuple("a", 1), "test5"),
    std::make_tuple(std::make_tuple("1243356", 0), "test6"),
    std::make_tuple(std::make_tuple("a1a1a1a1a1a1a1a1a1a1a1a1", 12), "test7"),
    std::make_tuple(std::make_tuple("!@345678&*()", 0), "test8"),
    std::make_tuple(std::make_tuple("aaaaaaaaaaaaaaaaaaaa", 20), "test9"),
    std::make_tuple(std::make_tuple("tatatatatatatatatatatatatatatatatatatatata", 42), "test10"),
    std::make_tuple(std::make_tuple("er11er11er11er11", 8), "test11"),
    std::make_tuple(std::make_tuple("eee___eee__", 6), "test12"),
    std::make_tuple(std::make_tuple("eee___eee__EEE", 9), "test13"),
    std::make_tuple(std::make_tuple("EEEE___EEEE", 8), "test14"),
    std::make_tuple(
        std::make_tuple(
            "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
            100),
        "test15"),
    std::make_tuple(std::make_tuple("фбсдуащуо", 0), "test16"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PapulinaYCountOfLettersMPI, InType>(kTestParam, PPC_SETTINGS_papulina_y_count_of_letters),
    ppc::util::AddFuncTask<PapulinaYCountOfLettersSEQ, InType>(kTestParam, PPC_SETTINGS_papulina_y_count_of_letters));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PapulinaYRunFuncTestsProcesses::PrintFuncTestName<PapulinaYRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, PapulinaYRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace papulina_y_count_of_letters
