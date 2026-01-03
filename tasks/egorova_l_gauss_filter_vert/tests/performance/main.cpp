#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kRows = 2000;
  static constexpr int kCols = 2000;
  static constexpr int kChannels = 3;

  void SetUp() override {
    input_.rows = kRows;
    input_.cols = kCols;
    input_.channels = kChannels;

    size_t total = static_cast<size_t>(kRows) * kCols * kChannels;

    input_.data.resize(total);
    for (size_t i = 0; i < total; ++i) {
      input_.data[i] = static_cast<uint8_t>((i * 7 + i / 31) % 256);
    }
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) final {
    return out.rows == kRows && out.cols == kCols && out.channels == kChannels &&
           out.data.size() == static_cast<size_t>(kRows) * kCols * kChannels;
  }

 private:
  InType input_;
};

TEST_P(EgorovaLGaussFilterVertRunPerfTests, GaussPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, EgorovaLGaussFilterVertMPI, EgorovaLGaussFilterVertSEQ>(
    PPC_SETTINGS_egorova_l_gauss_filter_vert);

INSTANTIATE_TEST_SUITE_P(EgorovaLGaussFilterPerf, EgorovaLGaussFilterVertRunPerfTests,
                         ppc::util::TupleToGTestValues(kPerfTasks), [](const auto &info) {
                           const auto &name = std::get<1>(info.param);
                           const auto run = std::get<2>(info.param);
                           return name + (run == ppc::performance::PerfResults::TypeOfRunning::kPipeline ? "_pipeline"
                                                                                                         : "_task");
                         });

}  // namespace egorova_l_gauss_filter_vert
