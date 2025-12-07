#include <gtest/gtest.h>
#include <mpi.h>

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkrebko_m_hypercube {

namespace {
bool ShouldSkipHypercubeTest() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size == 1) {
    return false;
  }

  bool skip_local = (world_size < 2 || (world_size & (world_size - 1)) != 0);

  int skip_int = skip_local ? 1 : 0;
  int skip_all = 0;
  MPI_Allreduce(&skip_int, &skip_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  return (skip_all == 1);
}
}  // namespace

class ShkrebkoMHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = {100, 1};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    bool value_ok = output_data.value == input_data_[0];
    bool finish_ok = output_data.finish == true;

    if (world_size == 1) {
      bool destination_ok = output_data.destination == 0;
      bool path_ok = output_data.path == std::vector<int>{0};
      return value_ok && finish_ok && destination_ok && path_ok;
    }

    return value_ok && finish_ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {0, 0};
};

TEST_P(ShkrebkoMHypercubePerfTests, RunPerfModes) {
  if (ShouldSkipHypercubeTest()) {
    return;
  }

  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    std::tuple_cat(ppc::util::MakeAllPerfTasks<InType, ShkrebkoMHypercubeMPI>(PPC_SETTINGS_shkrebko_m_hypercube),
                   ppc::util::MakeAllPerfTasks<InType, ShkrebkoMHypercubeSEQ>(PPC_SETTINGS_shkrebko_m_hypercube));

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrebkoMHypercubePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrebkoMHypercubePerfTests, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_hypercube
