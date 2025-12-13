#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"
#include "nikitina_v_trans_all_one_distrib/mpi/include/ops_mpi.hpp"
#include "nikitina_v_trans_all_one_distrib/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitina_v_trans_all_one_distrib {

class NikitinaVRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &param_info) {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param_info.param);
    auto task_type_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(param_info.param);
    return std::get<1>(params) + "_" + task_type_name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_ = std::vector<int>(size, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    if (output_data.size() != input_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] == 0) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(NikitinaVRunFuncTests, AllReduceSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "Size_10"), std::make_tuple(100, "Size_100"),
                                            std::make_tuple(123, "Size_123")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestParam, PPC_SETTINGS_nikitina_v_trans_all_one_distrib),
    ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_nikitina_v_trans_all_one_distrib));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(AllReduceTests, NikitinaVRunFuncTests, kGtestValues, NikitinaVRunFuncTests::PrintTestParam);

TEST(NikitinaVAllReduceMisc, RunWithEmptyVector) {
  std::vector<int> empty_vec;

  auto task_mpi = std::make_shared<TestTaskMPI>(empty_vec);
  ASSERT_EQ(task_mpi->GetStaticTypeOfTask(), ppc::task::TypeOfTask::kMPI);
  ASSERT_TRUE(task_mpi->Validation());
  task_mpi->PreProcessing();
  task_mpi->Run();
  task_mpi->PostProcessing();
  ASSERT_TRUE(task_mpi->GetOutput().empty());

  auto task_seq = std::make_shared<TestTaskSEQ>(empty_vec);
  ASSERT_EQ(task_seq->GetStaticTypeOfTask(), ppc::task::TypeOfTask::kSEQ);
  ASSERT_TRUE(task_seq->Validation());
  task_seq->PreProcessing();
  task_seq->Run();
  task_seq->PostProcessing();
  ASSERT_TRUE(task_seq->GetOutput().empty());
}

}  // namespace nikitina_v_trans_all_one_distrib
