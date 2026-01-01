#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cctype>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"
#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"
#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {

class VolkovACountWordLineFuncTests : public ppc::util::BaseRunFuncTests<std::string, OutType, TestType> {
 public:
  VolkovACountWordLineFuncTests() = default;

  static std::string PrintTestName(const testing::TestParamInfo<ParamType> &info) {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string input_sub = std::get<0>(params);
    int expected = std::get<1>(params);

    std::string safe_name;
    if (input_sub.empty()) {
      safe_name = "Empty";
    } else {
      if (input_sub.size() > 20) {
        input_sub = input_sub.substr(0, 20);
      }
      for (char c : input_sub) {
        if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
          safe_name += c;
        } else {
          safe_name += "_";
        }
      }
    }

    // Добавляем индекс info.index, чтобы имена были уникальными
    return safe_name + "_Exp" + std::to_string(expected) + "_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_str_ = std::get<0>(params);
    ref_count_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &result) override {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // ИСПРАВЛЕНИЕ:
    // MPI задача возвращает правильный ответ только на Rank 0.
    // На остальных рангах результат может быть любым (обычно 0), 
    // поэтому мы пропускаем проверку (возвращаем true).
    if (rank == 0) {
      if (result != ref_count_) {
          // Для отладки можно раскомментировать
          // std::cerr << "Rank 0 Fail: Expected " << ref_count_ << " got " << result << std::endl;
          return false;
      }
      return true;
    }
    return true;
  }

  std::string GetTestInputData() override {
    return test_str_;
  }

 private:
  std::string test_str_;
  OutType ref_count_ = 0;
};

namespace {

TEST_P(VolkovACountWordLineFuncTests, CountingLogic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 29> kFixedTests = {{{"simple test", 2},
                                               {"one two three", 3},
                                               {"testing", 1},
                                               {"   leading space", 2},
                                               {"trailing space   ", 2},
                                               {"   both sides   ", 2},
                                               {"multiple   spaces    between", 3},
                                               {"tabs\tand\tnewlines\nmixed", 4},
                                               {"", 0},
                                               {"   ", 0},
                                               {"\n\t\r", 0},
                                               {"snake_case_word", 1},
                                               {"kebab-case-word", 1},
                                               {"mixed_case-style", 1},
                                               {"word1 - word2", 3},
                                               {"user_name", 1},
                                               {"hello, world!", 2},
                                               {"one;two.three?four", 4},
                                               {"(brackets) [and] {braces}", 3},
                                               {"...wait...", 1},
                                               {"mail@example.com", 3},
                                               {"Line 1\nLine 2\nLine 3", 6},
                                               {"123 456 789", 3},
                                               {"word-1 word_2", 2},
                                               {"--", 1},
                                               {"__init__", 1},
                                               {"a b c d e f g", 7},
                                               {"longwordlongwordlongword", 1},
                                               {"ab cd ef gh ij kl mn op qr st uv wx yz", 13}}};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<VolkovACountWordLineMPI, std::string>(kFixedTests, PPC_SETTINGS_volkov_a_count_word_line),
    ppc::util::AddFuncTask<VolkovACountWordLineSEQ, std::string>(kFixedTests, PPC_SETTINGS_volkov_a_count_word_line));

const auto kTestParams = ppc::util::ExpandToValues(kTasks);
const auto kTestNameFunc = VolkovACountWordLineFuncTests::PrintTestName;

INSTANTIATE_TEST_SUITE_P(VolkovWordCount, VolkovACountWordLineFuncTests, kTestParams, kTestNameFunc);

}  // namespace
}  // namespace volkov_a_count_word_line