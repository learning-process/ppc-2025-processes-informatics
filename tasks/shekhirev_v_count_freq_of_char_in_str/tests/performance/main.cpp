#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/char_freq_mpi.hpp"
#include "../../seq/include/char_freq_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_char_freq_seq {

class ShekhirevVCharFreqPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const size_t size = 10000000;
    input_data_.str.resize(size, 'a');
    input_data_.str[size / 2] = 'b';
    input_data_.str[size - 1] = 'b';
    input_data_.target = 'b';
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0) {
      return output_data == 2;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(ShekhirevVCharFreqPerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, shekhirev_v_char_freq_mpi::CharFreqMPI, CharFreqSequential>(
        PPC_SETTINGS_shekhirev_v_count_freq_of_char_in_str);

INSTANTIATE_TEST_SUITE_P(CharFreqPerf, ShekhirevVCharFreqPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         ShekhirevVCharFreqPerfTests::CustomPerfTestName);

}  // namespace

}  // namespace shekhirev_v_char_freq_seq
