#include <gtest/gtest.h>
#include <mpi.h>

#include <fstream>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "lukin_i_torus_topology/mpi/include/ops_mpi.hpp"
#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lukin_i_torus_topology {
const int large_vector_size = 10'000'000;

class LukinIRunPerfTestProceses3 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_result_;

  const std::vector<int> dummy = {2, 0, 2, 6};
  std::vector<int> message;

  void SetUp() override {
    message = std::vector<int>(large_vector_size, 1);

    int proc_count = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (proc_count == 1) {
      input_data_ = std::make_tuple(0, 0, message);
      expected_result_ = std::make_tuple(std::vector<int>{}, message);
    } else if (proc_count == 2) {
      input_data_ = std::make_tuple(0, 1, message);
      expected_result_ = std::make_tuple(std::vector<int>{0, 1}, message);
    } else if (proc_count == 3) {
      input_data_ = std::make_tuple(0, 2, message);
      expected_result_ = std::make_tuple(std::vector<int>{0, 2}, message);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(0, 3, message);
      expected_result_ = std::make_tuple(std::vector<int>{0, 2, 3}, message);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> route = std::get<0>(output_data);
    if (route == dummy) {  // seq
      std::vector<int> recieved_message = std::get<1>(output_data);
      return recieved_message == message;
    } else {
      return expected_result_ == output_data;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LukinIRunPerfTestProceses3, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LukinIThorTopologyMPI, LukinIThorTopologySEQ>(
    PPC_SETTINGS_lukin_i_torus_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LukinIRunPerfTestProceses3::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, LukinIRunPerfTestProceses3, kGtestValues, kPerfTestName);

}  // namespace lukin_i_torus_topology
