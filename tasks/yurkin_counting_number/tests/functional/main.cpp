#include <gtest/gtest.h>

#include "util/include/func_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::TaskFunctionalTest<InType, OutType> {
 public:
  void SetUp() override {
    test_input_ = {'A', 'b', 'c', '1', '2', '3', '!', '!', '!', 'd', 'e', 'F', '#', '4', '5'};
  }

  bool CheckTestOutputData(const OutType &output) override {
    int cnt = 0;

    for (char c : test_input_) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        cnt++;
      }
    }

    return output == cnt;
  }
};

TEST_P(YurkinCountingNumberFuncTest, FunctionalRun) {
  Run();
}

INSTANTIATE_TEST_SUITE_P(FunctionalTest, YurkinCountingNumberFuncTest,
                         ppc::util::CreateTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSeq>(
                             PPC_SETTINGS_yurkin_counting_number));

}  // namespace yurkin_counting_number
