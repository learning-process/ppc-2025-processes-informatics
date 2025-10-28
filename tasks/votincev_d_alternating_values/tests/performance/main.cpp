#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatigValuesRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  const int kCount_ = 10000001;

  OutType expectedRes = 10000000;

  void SetUp() override {
    int sz = kCount_;
    std::vector<double> v;
    int swapper = 1;
    for (int i = 0; i < sz; i++) {
      v.push_back(i * swapper);  // 0 -1 2 -3 4 -5...
      swapper *= -1;
    }

    input_data_ = v;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // // для процессов 1,2, .. N-1
    // if(output_data == -1) {
    //   return true;
    // }
    // std::cout << (output_data == kCount_) << '\n';
    // 0й процесс должен вернуть верный результат
    // std::cout << "Inside CheckTestOutputData" <<output_data == kCount_ << "\n";
    return output_data == expectedRes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VotincevDAlternatingValuesMPI, VotincevDAlternatingValuesSEQ>(
        PPC_SETTINGS_votincev_d_alternating_values);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VotincevDAlternatigValuesRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDAlternatigValuesRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDAlternatigValuesRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_alternating_values
