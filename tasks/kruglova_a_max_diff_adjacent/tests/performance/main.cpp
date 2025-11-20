#include <gtest/gtest.h>

#include <cmath>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"
#include "kruglova_a_max_diff_adjacent/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"  // <-- PERF header

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int k_count = 500000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize(k_count);

    float acc = 0.0f;
    for (int i = 0; i < k_count; ++i) {
      float step = ((i % 5) - 2) * 0.7f + ((i % 3) - 1) * 0.3f;
      acc += step;
      input_data_[i] = acc;
    }

    expected_output_ = 0.0f;
    if (input_data_.size() > 1) {
      for (size_t i = 0; i + 1 < input_data_.size(); ++i) {
        float diff = std::abs(input_data_[i + 1] - input_data_[i]);
        if (diff > expected_output_) {
          expected_output_ = diff;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KruglovaAMaxDiffAdjacentPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KruglovaAMaxDiffAdjacentMPI, KruglovaAMaxDiffAdjacentSEQ>(
        PPC_SETTINGS_kruglova_a_max_diff_adjacent);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaAMaxDiffAdjacentPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaAMaxDiffAdjacentPerfTests, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_max_diff_adjacent
