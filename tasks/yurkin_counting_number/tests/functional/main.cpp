#include <array>
#include <cctype>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

  bool Validate(const InType &input, const OutType &output) override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int expected = std::get<0>(params);

    int count = 0;
    for (char c : input) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        count++;
      }
    }

    return count == expected && output == expected;
  }

  void PrepareInput() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_str = std::get<1>(params);

    InType input_data(input_str.begin(), input_str.end());
    SetInput(input_data);
  }
};

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, std::string("AbC123")),
                                            std::make_tuple(2, std::string("1a2B3")),
                                            std::make_tuple(4, std::string("AaBb123"))};

const auto kTasks =
    std::tuple_cat(ppc::util::FunctionalTasks<InType, OutType, YurkinCountingNumberFuncTests,
                                              yurkin_counting_number::YurkinCountingNumberSequential>::kValue,
                   ppc::util::FunctionalTasks<InType, OutType, YurkinCountingNumberFuncTests,
                                              yurkin_counting_number::YurkinCountingNumberParallel>::kValue);

INSTANTIATE_TEST_SUITE_P(YurkinCountingNumberFunc, ppc::util::ParametrizedFunctionalTest,
                         ::testing::Combine(::testing::ValuesIn(kTasks), ::testing::ValuesIn(kTestParam)),
                         YurkinCountingNumberFuncTests::PrintTestParam);

}  // namespace yurkin_counting_number
