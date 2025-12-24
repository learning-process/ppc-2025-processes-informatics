#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <ranges>  // NOLINT(misc-include-cleaner) - required for std::ranges::transform
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

  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &param_info) {
    auto params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(param_info.param);

    std::string text = std::get<0>(params);

    std::ranges::transform(text, text.begin(), [](unsigned char c) {
      if (std::isalnum(c) || c == '_') {
        return static_cast<char>(c);
      }
      return '_';
    });

    if (text.empty()) {
      text = "Empty";
    }

    return text + "_Exp" + std::to_string(std::get<1>(params)) + "_" + std::to_string(param_info.index);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return output_data == expected_output_;
    }
    return true;
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

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(std::string(""), 0),
    std::make_tuple(std::string("No terminators here"), 0),
    std::make_tuple(std::string("Sentence one. Sentence two! Sentence three?"), 3),
    std::make_tuple(std::string("Is this a question?"), 1),

    std::make_tuple(std::string("This is the end."), 1),
    std::make_tuple(std::string("Wow!"), 1),
    std::make_tuple(std::string("Wait, what?"), 1),

    std::make_tuple(std::string("Wait... What?! Stop!"), 3),
    std::make_tuple(std::string("Really?? No!"), 2),

    std::make_tuple(std::string("Test. Another one! Last?"), 3),
    std::make_tuple(std::string("Symbols? @ # $ %"), 1),
    std::make_tuple(std::string("!Start with terminator. End."), 3),

    std::make_tuple(std::string("The end is near..."), 1),
    std::make_tuple(std::string("A!B.C?"), 3),
    std::make_tuple(std::string(R"(Only terminators???!!!)"), 1),
    std::make_tuple(std::string("A..B.C"), 2),

    std::make_tuple(std::string("Ends with space after sentence. "), 1),
    std::make_tuple(std::string("Multiple spaces!   Next?  End."), 3),
    std::make_tuple(std::string("Mixed...???with!!!various??terminators."), 4),
    std::make_tuple(std::string("Single char?"), 1)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GusevDSentenceCountMPI, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count),
    ppc::util::AddFuncTask<GusevDSentenceCountSEQ, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevDSentenceCountFuncTests::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(SentenceCountBoundaryTests, GusevDSentenceCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gusev_d_sentence_count