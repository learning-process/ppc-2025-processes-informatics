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

#include "baldin_a_word_count/common/include/common.hpp"
#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"
#include "baldin_a_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_word_count {

class BaldinAWordCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string text = std::get<0>(test_param);

    for (char &c : text) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {
        c = '_';
      }
    }
    return text + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data = std::get<0>(params);
    expected_output = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output;
  }

  InType GetTestInputData() final {
    return input_data;
  }

 private:
  InType input_data;
  OutType expected_output;
};

namespace {

TEST_P(BaldinAWordCountFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
  // 1. Базовые случаи
  std::make_tuple(std::string("Hello world"), 2),
  std::make_tuple(std::string("One two  three   four"), 4),
  std::make_tuple(std::string("   Leading and trailing spaces   "), 4),
  std::make_tuple(std::string("Word-with-hyphen counted-as-one"), 2),
  std::make_tuple(std::string(""), 0),
  std::make_tuple(std::string("a"), 1),

  // 2. Пунктуация и символы
  std::make_tuple(std::string("Hello, world!"), 2),
  std::make_tuple(std::string("Wait... what?! Really?!"), 3),
  std::make_tuple(std::string("end-of-line; another_word, and.one and , one ; and ,one"), 8),
  std::make_tuple(std::string("Hello-world"), 1),

  // 3. Пробелы и переносы строк
  std::make_tuple(std::string("Line1\nLine2\n\nLine3"), 3),
  std::make_tuple(std::string("Tabs\tand spaces   mixed"), 4),
  std::make_tuple(std::string("   Multiple\n\twhitespace\nwords   "), 3),

  // 4. Большие буквы, цифры, смешанный регистр
  std::make_tuple(std::string("Hello WORLD mixed CASE"), 4),
  std::make_tuple(std::string("Numbers 123 and letters ABC123xyz"), 5),

  // 5. Много предложений
  std::make_tuple(std::string("This is sentence one. Here is two! And three? Yes."), 10)
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAWordCountMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_word_count),
                   ppc::util::AddFuncTask<BaldinAWordCountSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_word_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAWordCountFuncTests::PrintFuncTestName<BaldinAWordCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(WordCountTests, BaldinAWordCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_word_count
