#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

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

class ShkrebkoMHypercubeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int value = std::get<0>(test_params);

    input_data_ = {value, 1};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool value_ok = output_data.value == input_data_[0];
    bool finish_ok = output_data.finish == true;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

namespace {

TEST_P(ShkrebkoMHypercubeFuncTests, HypercubeRouting) {
  if (ShouldSkipHypercubeTest()) {
    MPI_Barrier(MPI_COMM_WORLD);
    return;
  }

  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "basic"), std::make_tuple(42, "medium"),
                                            std::make_tuple(100, "large")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShkrebkoMHypercubeMPI, InType>(kTestParam, PPC_SETTINGS_shkrebko_m_hypercube),
    ppc::util::AddFuncTask<ShkrebkoMHypercubeSEQ, InType>(kTestParam, PPC_SETTINGS_shkrebko_m_hypercube));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrebkoMHypercubeFuncTests::PrintFuncTestName<ShkrebkoMHypercubeFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShkrebkoMHypercube, ShkrebkoMHypercubeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkrebko_m_hypercube
