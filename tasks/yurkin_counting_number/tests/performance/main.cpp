#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {};

TEST_P(YurkinCountingNumberPerfTests, RunPerfModes) {
  RunAllPerf("yurkin_counting_number");
}

static const std::array<InType, 3> kInputs = {InType(1000, 'a'), InType(2000, 'b'), InType(5000, 'c')};

INSTANTIATE_TEST_SUITE_P(YurkinPerfTests, YurkinCountingNumberPerfTests, ::testing::ValuesIn(kInputs),
                         ::testing::PrintToStringParamName());

}  // namespace yurkin_counting_number
