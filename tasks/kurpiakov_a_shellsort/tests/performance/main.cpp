#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "kurpiakov_a_shellsort/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_shellsort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kPerfSize = 100000;
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    std::vector<int> vec(kPerfSize);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(-100000, 100000);
    for (int i = 0; i < kPerfSize; i++) {
      vec[i] = dist(gen);
    }
    input_data_ = std::make_tuple(kPerfSize, vec);

    expected_output_ = vec;
    std::sort(expected_output_.begin(), expected_output_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KurpiakovARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAShellsortMPI, KurpiakovAShellsortSEQ>(
    PPC_SETTINGS_kurpiakov_a_shellsort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KurpiakovARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_shellsort
