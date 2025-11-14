#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"
#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"
#include "sizov_d_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sizov_d_bubble_sort {

class SizovDRunPerfTestsBubbleSort : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t n = 10000;
    std::vector<int> data(n);
    for (std::size_t i = 0; i < n; ++i) {
      data[i] = static_cast<int>(n - i);
    }
    input_data_ = data;

    std::sort(data.begin(), data.end());
    expected_result_ = data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

TEST_P(SizovDRunPerfTestsBubbleSort, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, sizov_d_bubble_sort::SizovDBubbleSortMPI,
                                sizov_d_bubble_sort::SizovDBubbleSortSEQ>(PPC_SETTINGS_sizov_d_bubble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SizovDRunPerfTestsBubbleSort::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerf, SizovDRunPerfTestsBubbleSort, kGtestValues, kPerfTestName);

}  // namespace sizov_d_bubble_sort
