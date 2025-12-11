#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>

#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"  // NOLINT(misc-include-cleaner)
#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"  // NOLINT(misc-include-cleaner)
#include "util/include/perf_test_util.hpp"

namespace makovskiy_i_gauss_filter_vert {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;

class GaussFilterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {  // NOLINT(readability-convert-member-functions-to-static)
    constexpr int kWidth = 2000;
    constexpr int kHeight = 2000;
    std::vector<int> input_image(kWidth * kHeight);
    for (size_t idx = 0; idx < input_image.size(); ++idx) {
      input_image[idx] = static_cast<int>(idx % 256);
    }
    return std::make_tuple(std::move(input_image), kWidth, kHeight);
  }

  bool CheckTestOutputData(OutType &output_data) final {  // NOLINT(readability-convert-member-functions-to-static)
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      const auto &[input, width, height] = GetTestInputData();
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

const auto kSeqPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterSEQ>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kSeqGtestValues = ppc::util::TupleToGTestValues(kSeqPerfTasks);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfSEQ, GaussFilterPerfTests, kSeqGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)

const auto kMpiPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterMPI>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kMpiGtestValues = ppc::util::TupleToGTestValues(kMpiPerfTasks);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfMPI, GaussFilterPerfTests, kMpiGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)

}  // namespace
}  // namespace makovskiy_i_gauss_filter_vert
