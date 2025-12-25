#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "frolova_s_star_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_star_topology {

class FrolovaSRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kPerfDataSize = 2000000;
  InType input_data_ = 0;

  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &test_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);

    // Пропускаем SEQ тесты для топологии
    if (test_name.find("seq_enabled") != std::string::npos) {
      GTEST_SKIP() << "SEQ performance tests are skipped for topology tasks.";
    }

    input_data_ = kPerfDataSize;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0 && output_data <= kPerfDataSize;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FrolovaSRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSStarTopologyMPI, FrolovaSStarTopologySEQ>(
    PPC_SETTINGS_frolova_s_star_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaSRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FrolovaSRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace frolova_s_star_topology
