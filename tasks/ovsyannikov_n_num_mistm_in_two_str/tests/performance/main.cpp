#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int benchmark_size = 100000000;

    std::string sample_a(benchmark_size, 'a');
    std::string sample_b(benchmark_size, 'a');

    target_val_ = 0;
    for (int i = 0; i < benchmark_size; ++i) {
      if (i % 2 == 0) {
        sample_b[i] = 'b';
        target_val_++;
      }
    }

    bench_data_ = std::make_pair(sample_a, sample_b);
  }

  bool CheckTestOutputData(OutType &calculated_res) final {
    return calculated_res == target_val_;
  }

  InType GetTestInputData() final {
    return bench_data_;
  }

 private:
  InType bench_data_;
  OutType target_val_ = 0;
};

TEST_P(OvsyannikovNRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kPerfTaskList =
    ppc::util::MakeAllPerfTasks<InType, OvsyannikovNNumMistmInTwoStrMPI, OvsyannikovNNumMistmInTwoStrSEQ>(
        PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str);

const auto kTestParams = ppc::util::TupleToGTestValues(kPerfTaskList);

const auto kCustomName = OvsyannikovNRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OvsyannikovNRunPerfTestProcesses, kTestParams, kCustomName);

}  // namespace ovsyannikov_n_num_mistm_in_two_str
