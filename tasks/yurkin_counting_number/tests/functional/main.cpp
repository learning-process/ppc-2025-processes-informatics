#include <array>
#include <cctype>
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
    return std::get<0>(test_param) + std::string("_") + std::get<1>(test_param);
  }

  bool CheckTestOutputData(const OutType &output_data) override {
    const auto &test_param = GetParam();
    const int expected = std::get<0>(test_param);
    return expected == output_data;
  }

  InType GetTestInputData() override {
    const auto &test_param = GetParam();
    const std::string &s = std::get<1>(test_param);

    InType v(s.begin(), s.end());
    return v;
  }
};

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, std::string("AbC123")),
                                            std::make_tuple(0, std::string("12345")),
                                            std::make_tuple(5, std::string("aAzZb!!!"))};

const auto kTasks = std::tuple_cat(
    ppc::util::MakeTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(PPC_SETTINGS_yurkin_counting_number),
    ppc::util::MakeTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
        PPC_SETTINGS_yurkin_counting_number));

INSTANTIATE_TEST_SUITE_P(FunctionalRun, YurkinCountingNumberFuncTests,
                         ::testing::Combine(::testing::ValuesIn(kTestParam), ppc::util::TupleToGTestValues(kTasks)),
                         YurkinCountingNumberFuncTests::PrintTestParam);

}  // namespace yurkin_counting_number
