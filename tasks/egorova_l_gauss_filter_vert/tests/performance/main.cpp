#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  static constexpr int kRows = 2000;
  static constexpr int kCols = 2000;
  static constexpr int kChannels = 1;
  InType input_data_;

 public:
  void SetUp() override {
    input_data_.rows = kRows;
    input_data_.cols = kCols;
    input_data_.channels = kChannels;
    input_data_.data.assign(
        static_cast<std::size_t>(kRows) * static_cast<size_t>(kCols) * static_cast<size_t>(kChannels), 128);
  }

  bool CheckTestOutputData(OutType& output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    return output_data.rows == kRows && output_data.cols == kCols &&
           output_data.data.size() ==
               static_cast<std::size_t>(kRows) * static_cast<size_t>(kCols) * static_cast<size_t>(kChannels);
  }

  InType GetTestInputData() final { return input_data_; }
};

TEST_P(EgorovaLGaussFilterVertRunPerfTests, EgorovaLGaussPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, EgorovaLGaussFilterVertMPI, EgorovaLGaussFilterVertSEQ>(
    PPC_SETTINGS_egorova_l_gauss_filter_vert);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = EgorovaLGaussFilterVertRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(EgorovaLGaussPerf, EgorovaLGaussFilterVertRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace egorova_l_gauss_filter_vert