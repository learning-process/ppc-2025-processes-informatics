#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <random>
#include <vector>

#include "nikitina_v_quick_sort_merge/common/include/common.hpp"
#include "nikitina_v_quick_sort_merge/mpi/include/ops_mpi.hpp"
#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitina_v_quick_sort_merge {

class RunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int count = 1000000;
    input_data_.resize(count);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(-100000, 100000);
    for (int &val : input_data_) {
      val = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(RunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_nikitina_v_quick_sort_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = RunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(QuickSortPerfTests, RunPerfTests, kGtestValues, kPerfTestName);  // NOLINT(cert-err58-cpp)

}  // namespace nikitina_v_quick_sort_merge
