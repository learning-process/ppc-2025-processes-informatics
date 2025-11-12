#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "krykov_e_word_count/common/include/common.hpp"
#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"
#include "krykov_e_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_word_count {

class KrykovEWordCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KrykovEWordCountFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    std::string text = std::get<0>(test_param);
    return text + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KrykovEWordCountFuncTests, WordCountTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {
    std::make_tuple(std::string("Hello world"), 2),
    std::make_tuple(std::string("One two  three   four"), 4),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KrykovEWordCountMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count),
                   ppc::util::AddFuncTask<KrykovEWordCountSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovEWordCountFuncTests::PrintFuncTestName<KrykovEWordCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(WordCountTests, KrykovEWordCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krykov_e_word_count
