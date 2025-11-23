#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "gusev_d_sentence_count/common/include/common.hpp"
#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"
#include "gusev_d_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_sentence_count {

class GusevDSentenceCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  GusevDSentenceCountFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    std::string text = std::get<0>(test_param);
    std::replace(text.begin(), text.end(), ' ', '_');
    return text + "_Expected_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(GusevDSentenceCountFuncTests, SentenceBoundaryTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(std::string(""), 0),                                     
    std::make_tuple(std::string("No terminators here"), 0),                 
    std::make_tuple(std::string("Sentence one. Sentence two! Sentence three?"), 3),
    std::make_tuple(std::string("Is this a question?"), 1),                  

    std::make_tuple(std::string("This is the end."), 1),                     
    std::make_tuple(std::string("Wow!"), 1),                                
    std::make_tuple(std::string("Wait, what?"), 1),                         

    std::make_tuple(std::string("Wait... What?! Stop!"), 2),
    std::make_tuple(std::string("Really?? No!"), 2),              

    std::make_tuple(std::string("Test. Another one! Last?"), 3),
    std::make_tuple(std::string("Symbols? @ # $ %"), 1),                     
    std::make_tuple(std::string("!Start with terminator. End."), 2),          
    
    std::make_tuple(std::string("The end is near..."), 1),                    
    std::make_tuple(std::string("A!B.C?"), 3),                          
    std::make_tuple(std::string("Only terminators???!!!"), 1),               
    std::make_tuple(std::string("A..B.C"), 2)                               
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GusevDSentenceCountMPI, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count),
                   ppc::util::AddFuncTask<GusevDSentenceCountSEQ, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevDSentenceCountFuncTests::PrintFuncTestName<GusevDSentenceCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(SentenceCountBoundaryTests, GusevDSentenceCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gusev_d_sentence_count