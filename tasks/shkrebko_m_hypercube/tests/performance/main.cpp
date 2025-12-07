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
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    bool value_ok = output_data.value == input_data_[0];
    bool finish_ok = output_data.finish == true;

    if (world_rank == 0) {
      bool path_not_empty = !output_data.path.empty();
      bool path_starts_at_zero = output_data.path.front() == 0;
      bool path_ends_at_dest = output_data.path.back() == input_data_[1];

      return value_ok && path_not_empty && finish_ok && path_starts_at_zero && path_ends_at_dest;
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
    MPI_Barrier(MPI_COMM_WORLD);
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
