#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "maslova_u_fast_sort_simple/mpi/include/ops_mpi.hpp"
#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_fast_sort_simple {

class MaslovaUFastSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const size_t count = 1000000;
    input_data_.resize(count);
    std::mt19937 gen(42);
    for (size_t i = 0; i < count; ++i) {
        input_data_[i] = static_cast<int>(gen() % 100000);
    }
    expected_output_ = input_data_;
    std::sort(expected_output_.begin(), expected_output_.end());
  }
  InType GetTestInputData() final { return input_data_; }
  bool CheckTestOutputData(OutType &output_data) final { return output_data == expected_output_; }
 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(MaslovaUFastSortPerfTests, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MaslovaUFastSortSimpleMPI, MaslovaUFastSortSimpleSEQ>(
    PPC_SETTINGS_maslova_u_fast_sort_simple);

INSTANTIATE_TEST_SUITE_P(fastSortPerf, MaslovaUFastSortPerfTests, 
                         ppc::util::TupleToGTestValues(kAllPerfTasks), 
                         MaslovaUFastSortPerfTests::CustomPerfTestName);

}  // namespace maslova_u_fast_sort_simple