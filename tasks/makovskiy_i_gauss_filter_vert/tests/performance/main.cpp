#include <gtest/gtest.h>
#include <vector>
#include <mpi.h>

#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"

namespace makovskiy_i_gauss_filter_vert {

class GaussFilterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    constexpr int width = 2000;
    constexpr int height = 2000;
    std::vector<int> input_image(width * height);
    for (size_t i = 0; i < input_image.size(); ++i) {
        input_image[i] = (i % 256);
    }
    return std::make_tuple(std::move(input_image), width, height);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        const auto& [input, width, height] = GetTestInputData();
        return output_data.size() == static_cast<size_t>(width * height);
    }
    return true;
  }
};

TEST_P(GaussFilterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kPerfTestName = GaussFilterPerfTests::CustomPerfTestName;

const auto kSeqPerfTasks = ppc::util::MakeAllPerfTasks<InType, GaussFilterSEQ>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kSeqGtestValues = ppc::util::TupleToGTestValues(kSeqPerfTasks);
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfSEQ, GaussFilterPerfTests, kSeqGtestValues, kPerfTestName);

const auto kMpiPerfTasks = ppc::util::MakeAllPerfTasks<InType, GaussFilterMPI>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kMpiGtestValues = ppc::util::TupleToGTestValues(kMpiPerfTasks);
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfMPI, GaussFilterPerfTests, kMpiGtestValues, kPerfTestName);

}  // namespace
}  // namespace makovskiy_i_gauss_filter_vert