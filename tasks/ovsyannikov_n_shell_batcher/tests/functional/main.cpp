#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "ovsyannikov_n_shell_batcher/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<TestType>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(static_cast<double>(output_data[i]) - static_cast<double>(expected_[i])) > 1e-6) {
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
  OutType expected_;
};

TEST_P(OvsyannikovNShellBatcherFuncTest, TestSorting) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {{
    std::make_tuple(std::vector<int>{}, std::vector<int>{}, "EmptyVector"),
    std::make_tuple(std::vector<int>{5}, std::vector<int>{5}, "SingleElement"),
    std::make_tuple(std::vector<int>{3, 1, 2}, std::vector<int>{1, 2, 3}, "SmallRandom"),
    std::make_tuple(std::vector<int>{10, 9, 8, 7, 6}, std::vector<int>{6, 7, 8, 9, 10}, "ReverseSorted"),
    std::make_tuple(std::vector<int>{1, 2, 3, 4}, std::vector<int>{1, 2, 3, 4}, "AlreadySorted"),
    std::make_tuple(std::vector<int>{2, 2, 1, 1}, std::vector<int>{1, 1, 2, 2}, "Duplicates"),
}};

INSTANTIATE_TEST_SUITE_P(
    ovsyannikov_n_shell_batcher_mpi, OvsyannikovNShellBatcherFuncTest,
    ppc::util::ExpandToValues(ppc::util::AddFuncTask<OvsyannikovNShellBatcherMPI, InType>(kTestParam, "mpi")),
    [](const testing::TestParamInfo<OvsyannikovNShellBatcherFuncTest::ParamType> &info) {
      return OvsyannikovNShellBatcherFuncTest::PrintFuncTestName<OvsyannikovNShellBatcherFuncTest>(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ovsyannikov_n_shell_batcher_seq, OvsyannikovNShellBatcherFuncTest,
    ppc::util::ExpandToValues(ppc::util::AddFuncTask<OvsyannikovNShellBatcherSEQ, InType>(kTestParam, "seq")),
    [](const testing::TestParamInfo<OvsyannikovNShellBatcherFuncTest::ParamType> &info) {
      return OvsyannikovNShellBatcherFuncTest::PrintFuncTestName<OvsyannikovNShellBatcherFuncTest>(info);
    });
}  // namespace ovsyannikov_n_shell_batcher
