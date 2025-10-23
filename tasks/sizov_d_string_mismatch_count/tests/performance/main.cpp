#include <gtest/gtest.h>

#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"
#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sizov_d_string_mismatch_count {

class SizovRunPerfTestsMismatch : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    std::string a(1'000'000, 'a');
    std::string b = a;

    for (std::size_t i = 0; i < a.size() / 10; ++i) {
      b[i] = 'b';
    }

    expected_result_ = static_cast<int>(a.size() / 10);
    input_data_ = std::make_tuple(std::move(a), std::move(b));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType& output_data) final {
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StringMismatchCountParallelMPI, StringMismatchCountSequential>(
        PPC_SETTINGS_sizov_d_string_mismatch_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SizovRunPerfTestsMismatch::CustomPerfTestName;

TEST_P(SizovRunPerfTestsMismatch, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, SizovRunPerfTestsMismatch, kGtestValues, kPerfTestName);

}  // namespace sizov_d_string_mismatch_count
