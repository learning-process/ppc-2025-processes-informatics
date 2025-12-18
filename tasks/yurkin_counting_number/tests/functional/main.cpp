#include <gtest/gtest.h>

#include "core/include/task_test.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::core::TaskTest<InType, OutType> {
 public:
  void SetUp() override {
    input_data_ = {'A', 'b', 'c', '1', '2', '3', 'd', 'e', 'F', '#'};
  }

  bool CheckTestOutput(const OutType &output) override {
    int cnt = 0;
    for (char c : input_data_) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        cnt++;
      }
    }
    return cnt == output;
  }
};

TEST_P(YurkinCountingNumberFuncTest, FunctionalRun) {
  Run();
}

INSTANTIATE_TEST_SUITE_P(Functional, YurkinCountingNumberFuncTest,
                         testing::Values(std::make_shared<YurkinCountingNumberSeq>(),
                                         std::make_shared<YurkinCountingNumberMPI>()));

}  // namespace yurkin_counting_number
