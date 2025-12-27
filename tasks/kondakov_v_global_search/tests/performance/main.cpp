#include <gtest/gtest.h>

#include <cmath>

#include "kondakov_v_global_search/common/include/common.hpp"
#include "kondakov_v_global_search/mpi/include/ops_mpi.hpp"
#include "kondakov_v_global_search/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kondakov_v_global_search {

class KondakovVGlobalSearchRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_.func = [](double x) { return ((x - 3.14) * (x - 3.14)) + (0.1 * std::sin(50.0 * x)); };
    input_data_.left = 0.0;
    input_data_.right = 6.28;
    input_data_.accuracy = 1e-7;
    input_data_.reliability = 2.0;
    input_data_.max_iterations = 15000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data.argmin) && std::isfinite(output_data.value) && output_data.argmin >= 0.0 &&
           output_data.argmin <= 6.28;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KondakovVGlobalSearchRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KondakovVGlobalSearchMPI, KondakovVGlobalSearchSEQ>(
    PPC_SETTINGS_kondakov_v_global_search);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondakovVGlobalSearchRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KondakovVGlobalSearchTests, KondakovVGlobalSearchRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace kondakov_v_global_search
