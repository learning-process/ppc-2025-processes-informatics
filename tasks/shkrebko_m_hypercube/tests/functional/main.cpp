#include <gtest/gtest.h>

#include <vector>
#include <mpi.h>
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
    return output_data.value == input_data_[0] && !output_data.path.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {0, 0};
};

TEST_P(ShkrebkoMHypercubeFuncTests, HypercubeRouting) {
  // Для MPI версии пропускаем тест, если число процессов не соответствует гиперкубу
  auto task_info = GetParam();
  std::string task_name = std::get<1>(task_info);
  
  if (task_name == "shkrebko_m_hypercube_mpi_enabled") {
    if (ShouldSkipHypercubeTest()) {
      GTEST_SKIP() << "Skipping hypercube test for non-power-of-two process count or process count < 2";
    }
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

}  // namespace shkrebko_m_hypercube