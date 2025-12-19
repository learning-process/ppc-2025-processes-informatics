#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BasePerfTest<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const std::string &str_input = std::get<1>(params);

    input_data_.assign(str_input.begin(), str_input.end());
  }

  InType GetPerfInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 3> kPerfParam = {std::make_tuple(0, std::string(2000000, 'A')),
                                            std::make_tuple(1, std::string(3000000, 'b')),
                                            std::make_tuple(2, std::string(4000000, '1'))};

const auto kTasksPerf = std::tuple_cat(
    ppc::util::AddPerfTask<YurkinCountingNumberMPI, InType>(kPerfParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddPerfTask<YurkinCountingNumberSEQ, InType>(kPerfParam, PPC_SETTINGS_yurkin_counting_number));

const auto kPerfValues = ppc::util::ExpandToValues(kTasksPerf);

const auto kPerfName = YurkinCountingNumberPerfTests::PrintPerfTestName<YurkinCountingNumberPerfTests>;

TEST_P(YurkinCountingNumberPerfTests, PerfTest) {
  ExecutePerfTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(YurkinPerfTests, YurkinCountingNumberPerfTests, kPerfValues, kPerfName);

}  // namespace

}  // namespace yurkin_counting_number
