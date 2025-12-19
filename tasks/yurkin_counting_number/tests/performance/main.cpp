#include <string>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  InType input_data_;

  void PrepareInput(std::size_t n) override {
    input_data_.clear();
    input_data_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      input_data_.push_back('A');
    }
  }

  InType &GetInput() override {
    return input_data_;
  }
};

const auto kTasks = std::make_tuple(ppc::util::PerfTask<InType, OutType, YurkinCountingNumberPerfTests,
                                                        yurkin_counting_number::YurkinCountingNumberSEQ>(),
                                    ppc::util::PerfTask<InType, OutType, YurkinCountingNumberPerfTests,
                                                        yurkin_counting_number::YurkinCountingNumberMPI>());

INSTANTIATE_TEST_SUITE_P(YurkinCountingNumberPerf, ppc::util::ParametrizedPerformanceTest,
                         ::testing::Combine(::testing::ValuesIn(kTasks), ::testing::Values(100000)));

}  // namespace yurkin_counting_number
