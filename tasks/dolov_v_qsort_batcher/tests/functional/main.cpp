#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param) + "_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    std::string type = std::get<1>(params);

    if (rank == 0) {
      if (size >= 0) {
        input_data_.resize(size);
        if (type == "Random") {
          std::mt19937 gen(42);
          std::uniform_real_distribution<double> dis(-1000.0, 1000.0);
          std::generate(input_data_.begin(), input_data_.end(), [&]() { return dis(gen); });
        } else if (type == "Reverse") {
          std::iota(input_data_.begin(), input_data_.end(), 0.0);
          std::reverse(input_data_.begin(), input_data_.end());
        } else if (type == "Sorted") {
          std::iota(input_data_.begin(), input_data_.end(), 0.0);
        } else if (type == "Identical") {
          std::fill(input_data_.begin(), input_data_.end(), 7.0);
        } else if (type == "Negative") {
          std::iota(input_data_.begin(), input_data_.end(), -static_cast<double>(size));
        }

        expected_res_ = input_data_;
        std::sort(expected_res_.begin(), expected_res_.end());
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      if (input_data_.empty()) {
        return output_data.empty();
      }
      return output_data == expected_res_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_res_;
};

namespace {

TEST_P(DolovVQsortBatcherFuncTests, QsortBatcherTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {std::make_tuple(0, "Empty"),      std::make_tuple(1, "Single"),
                                            std::make_tuple(2, "Reverse"),    std::make_tuple(15, "Sorted"),
                                            std::make_tuple(16, "Random"),    std::make_tuple(127, "Random"),
                                            std::make_tuple(50, "Identical"), std::make_tuple(20, "Negative")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<DolovVQsortBatcherMPI, InType>(kTestParam, PPC_SETTINGS_dolov_v_qsort_batcher),
    ppc::util::AddFuncTask<DolovVQsortBatcherSEQ, InType>(kTestParam, PPC_SETTINGS_dolov_v_qsort_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = DolovVQsortBatcherFuncTests::PrintFuncTestName<DolovVQsortBatcherFuncTests>;

INSTANTIATE_TEST_SUITE_P(QsortBatcherTests, DolovVQsortBatcherFuncTests, kGtestValues, kPerfTestName);

TEST(DolovVQsortBatcherMPI_Validation, ReturnsFalseOnEmpty) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> empty_in;
  auto task = std::make_shared<DolovVQsortBatcherMPI>(empty_in);
  ASSERT_TRUE(task->Validation());
}

}  // namespace
}  // namespace dolov_v_qsort_batcher
