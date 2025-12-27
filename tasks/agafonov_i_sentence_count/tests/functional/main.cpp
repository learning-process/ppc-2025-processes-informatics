#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "agafonov_i_sentence_count/common/include/common.hpp"
#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"
#include "agafonov_i_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace agafonov_i_sentence_count {

class SentenceCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_case_id_ = std::get<0>(params);

    switch (test_case_id_) {
      case 1:
        input_data_ = "Hello world. This is a test. Another sentence! And one more? Yes.";
        expected_output_ = 5;
        break;
      case 2:
        input_data_ = "Simple text without punctuation";
        expected_output_ = 1;
        break;
      case 3:
        input_data_ = "First. Second. Third.";
        expected_output_ = 3;
        break;
      case 4:
        input_data_ = "...";
        expected_output_ = 0;
        break;
      case 5:
        input_data_ = "A. B. C. D.";
        expected_output_ = 4;
        break;
      default:
        input_data_ = "Default test case.";
        expected_output_ = 1;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int test_case_id_ = 0;
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(SentenceCountFuncTests, RunSentenceCountFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {
    std::make_tuple(1, "five_sentences"), std::make_tuple(2, "one_sentence_no_punctuation"),
    std::make_tuple(3, "three_sentences"), std::make_tuple(4, "only_dots"), std::make_tuple(5, "four_short_sentences")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SentenceCountMPI, InType>(kTestParams, PPC_SETTINGS_agafonov_i_sentence_count),
    ppc::util::AddFuncTask<SentenceCountSEQ, InType>(kTestParams, PPC_SETTINGS_agafonov_i_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SentenceCountFuncTests::PrintFuncTestName<SentenceCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(SentenceCountFuncTests, SentenceCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace agafonov_i_sentence_count
