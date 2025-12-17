#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    // Берём параметр теста
    const auto &param = GetParam();
    TestType test = std::get<2>(param);

    // В SetUp подготавливаем строку, буквы и цифры
    const std::string &s = std::get<1>(test);
    input_data_.assign(s.begin(), s.end());
  }

  // Всё остальное не трогаем
 private:
  InType input_data_{};
};

namespace {

TEST_P(YurkinCountingNumberFuncTest, CountLetters) {
  ExecuteTest(GetParam());
}

// Примеры тестовых строк: буквы + цифры
const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "abc123"), std::make_tuple(5, "A1B2C3D4E"),
                                            std::make_tuple(7, "xyzXYZ1234567")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const char *kPerfTestName = "YurkinCountLetters";

INSTANTIATE_TEST_SUITE_P(CountLettersTests, YurkinCountingNumberFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yurkin_counting_number
