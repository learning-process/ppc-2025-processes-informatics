#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>

#include "gusev_d_radix_double/common/include/common.hpp"
#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gusev_d_radix_double {

class GusevDRadixDoublePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static std::string CustomPerfTestName(
      const testing::TestParamInfo<ppc::util::PerfTestParam<InType, OutType>> &param_info) {
    return std::to_string(param_info.index);
  }

  void SetUp() override {
    size_t count = 1000000; 
    input_data_ = std::vector<double>(count);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-10000.0, 10000.0);
    
    for (size_t i = 0; i < count; ++i) {
      input_data_[i] = dis(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
     return std::is_sorted(output_data.begin(), output_data.end());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(GusevDRadixDoublePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GusevDRadixDoubleMPI, GusevDRadixDoubleSEQ>(PPC_SETTINGS_gusev_d_radix_double);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevDRadixDoublePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevDRadixDoublePerfTests, kGtestValues, kPerfTestName);

}  // namespace gusev_d_radix_double