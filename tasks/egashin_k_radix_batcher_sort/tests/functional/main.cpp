#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"
#include "egashin_k_radix_batcher_sort/mpi/include/ops_mpi.hpp"
#include "egashin_k_radix_batcher_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_radix_batcher_sort {

class EgashinKRadixBatcherSortFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(param);
    expected_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output) override {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    if (output.size() != expected_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
      if (output[i] != expected_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() override {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

namespace {

// Helper function to generate sorted output from input
inline OutType SortedOutput(const InType &input) {
  OutType result = input;
  std::ranges::sort(result);
  return result;
}

// Test cases with unique names
const std::array<TestType, 15> kTests = {
    std::make_tuple(InType{}, OutType{}, "EmptyArray"),
    std::make_tuple(InType{42.0}, OutType{42.0}, "SingleElement"),
    std::make_tuple(InType{1.0, 2.0}, OutType{1.0, 2.0}, "TwoElementsSorted"),
    std::make_tuple(InType{2.0, 1.0}, OutType{1.0, 2.0}, "TwoElementsReverse"),
    std::make_tuple(InType{5.5, 2.2, 8.8, 1.1, 9.9, 3.3}, OutType{1.1, 2.2, 3.3, 5.5, 8.8, 9.9}, "MultiplePositive"),
    std::make_tuple(InType{-1.0, -5.0, -2.0, -8.0, -3.0}, OutType{-8.0, -5.0, -3.0, -2.0, -1.0}, "MultipleNegative"),
    std::make_tuple(InType{3.0, -1.0, 4.0, -5.0, 2.0, -3.0}, OutType{-5.0, -3.0, -1.0, 2.0, 3.0, 4.0}, "MixedSigns"),
    std::make_tuple(InType{0.0, -0.0, 1.0, -1.0}, SortedOutput({0.0, -0.0, 1.0, -1.0}), "WithZeros"),
    std::make_tuple(InType{3.14, 2.72, 1.41, 1.62}, OutType{1.41, 1.62, 2.72, 3.14}, "DecimalNumbers"),
    std::make_tuple(InType{1e10, 1e5, 1e15, 1e1, 1e8}, OutType{1e1, 1e5, 1e8, 1e10, 1e15}, "LargeNumbers"),
    std::make_tuple(InType{1e-10, 1e-5, 1e-15, 1e-1, 1e-8}, OutType{1e-15, 1e-10, 1e-8, 1e-5, 1e-1}, "SmallNumbers"),
    std::make_tuple(InType{5.0, 2.0, 5.0, 2.0, 5.0}, OutType{2.0, 2.0, 5.0, 5.0, 5.0}, "Duplicates"),
    std::make_tuple(InType{1.0, 2.0, 3.0, 4.0, 5.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0}, "AlreadySorted"),
    std::make_tuple(InType{5.0, 4.0, 3.0, 2.0, 1.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0}, "ReverseSorted"),
    std::make_tuple(InType{8.0, 4.0, 2.0, 6.0, 1.0, 5.0, 7.0, 3.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    "PowerOf2Size")};

TEST_P(EgashinKRadixBatcherSortFuncTest, Sorting) {
  ExecuteTest(GetParam());
}

const auto kTaskParams =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTests, PPC_SETTINGS_egashin_k_radix_batcher_sort),
                   ppc::util::AddFuncTask<TestTaskMPI, InType>(kTests, PPC_SETTINGS_egashin_k_radix_batcher_sort));

INSTANTIATE_TEST_SUITE_P(EgashinKRadixBatcherSortFunc, EgashinKRadixBatcherSortFuncTest, ppc::util::ExpandToValues(kTaskParams));

}  // namespace

}  // namespace egashin_k_radix_batcher_sort
