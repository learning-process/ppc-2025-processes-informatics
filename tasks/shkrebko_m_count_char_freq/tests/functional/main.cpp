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

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_count_char_freq {

class ShkrebkoMCountCharFreqFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
 void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(param);
    
    switch (test_id) {
      case 1:
        input_data_ = std::make_pair("Alolo polo", 'l');
        expected_data_ = 3;
        break;
      case 2:
        input_data_ = std::make_pair("aramopma", 'm');
        expected_data_ = 2;
        break;
      case 3:
        input_data_ = std::make_pair("banana", 'a');
        expected_data_ = 3;
        break;
      case 4:
        input_data_ = std::make_pair("abcde", 'z');
        expected_data_ = 0;
        break;
    }
  }

   bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(ShkrebkoMCountCharFreqFuncTests, CountCharFrequency) {
  ExecuteTest(GetParam());
}


const std::array<TestType, 4> kTestParam = {
    std::make_tuple(1, "test1"),      
    std::make_tuple(2, "test2"),      
    std::make_tuple(3, "test3"),      
    std::make_tuple(4, "test4"),            
};

const auto kTestTasksList = 
    std::tuple_cat(ppc::util::AddFuncTask<shkrebko_m_count_char_freq::ShkrebkoMCountCharFreqMPI, InType>(
                       kTestParam, PPC_SETTINGS_shkrebko_m_count_char_freq),
                   ppc::util::AddFuncTask<shkrebko_m_count_char_freq::ShkrebkoMCountCharFreqSEQ, InType>(
                       kTestParam, PPC_SETTINGS_shkrebko_m_count_char_freq));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrebkoMCountCharFreqFuncTests::PrintFuncTestName<ShkrebkoMCountCharFreqFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShkrebkoMCharFreq, ShkrebkoMCountCharFreqFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkrebko_m_count_char_freq