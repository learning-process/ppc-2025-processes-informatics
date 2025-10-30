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

#include <ostream>

#include "krykov_e_word_count/common/include/common.hpp"
#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"
#include "krykov_e_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_word_count {

class KrykovEWordCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    static std::vector<std::string> test_lines = ReadTestLines();


    (void)std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_count_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_count_;

  std::vector<std::string> ReadTestLines() {
    std::vector<std::string> lines;
    
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_krykov_e_word_count, "lines.txt");
    std::ifstream file(abs_path);
      
      
      
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && (line.back() == '\r')||(line.back() == '\n')||(line.back() == '\t')) {
        line.pop_back();
      }
      lines.push_back(line);
    }
      
    file.close();
      
    //std::cout << "Loaded " << lines.size() << " test lines from file" << std::endl;
      
    
    return lines;
  }

  int CalculateExpectedWordCount(const std::string& text) {
    if (text.empty()) return 0;

    int count = 0;
    bool in_word = false;

    for (char c : text) {
      if (std::isspace(c) || std::ispunct(c)) {
        if (in_word) {
          count++;
          in_word = false;
        }
      } else {
        in_word = true;
      }
    }

    if (in_word) {
      count++;
    }

    return count;
  }
};

namespace {

TEST_P(KrykovEWordCountFuncTests, WordCountTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::string("default")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KrykovEWordCountMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count),
                   ppc::util::AddFuncTask<KrykovEWordCountSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovEWordCountFuncTests::PrintFuncTestName<KrykovEWordCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(WordCountTests, KrykovEWordCountFuncTests, kGtestValues, kPerfTestName);


}  // namespace

}  // namespace krykov_e_word_count
