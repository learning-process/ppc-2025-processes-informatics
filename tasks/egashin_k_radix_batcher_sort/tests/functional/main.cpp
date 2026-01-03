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

class EgashinKRunFuncTestsRadixBatcherSort : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_data_[i]) {
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
  OutType expected_data_;
};

namespace {

inline OutType SortedOutput(const InType &input) {
  OutType result = input;
  std::ranges::sort(result);
  return result;
}

const std::array<TestType, 30> kTestParam = {
    std::make_tuple(InType{}, OutType{}, "EmptyArray"),
    std::make_tuple(InType{42.0}, OutType{42.0}, "SinglePositive"),
    std::make_tuple(InType{-42.0}, OutType{-42.0}, "SingleNegative"),
    std::make_tuple(InType{0.0}, OutType{0.0}, "SingleZero"),
    std::make_tuple(InType{1.0, 2.0}, OutType{1.0, 2.0}, "TwoElementsSorted"),
    std::make_tuple(InType{2.0, 1.0}, OutType{1.0, 2.0}, "TwoElementsReverse"),
    std::make_tuple(InType{-2.0, -1.0}, OutType{-2.0, -1.0}, "TwoNegativeSorted"),
    std::make_tuple(InType{-1.0, -2.0}, OutType{-2.0, -1.0}, "TwoNegativeReverse"),
    std::make_tuple(InType{3.0, 1.0, 2.0}, OutType{1.0, 2.0, 3.0}, "ThreeElements"),
    std::make_tuple(InType{-3.0, -1.0, -2.0}, OutType{-3.0, -2.0, -1.0}, "ThreeNegative"),
    std::make_tuple(InType{5.0, 3.0, 1.0, 4.0, 2.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0}, "FiveElements"),
    std::make_tuple(InType{7.0, 3.0, 5.0, 1.0, 6.0, 2.0, 4.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                    "SevenElements"),
    std::make_tuple(InType{5.5, 2.2, 8.8, 1.1, 9.9, 3.3}, OutType{1.1, 2.2, 3.3, 5.5, 8.8, 9.9}, "MultiplePositive"),
    std::make_tuple(InType{-1.0, -5.0, -2.0, -8.0, -3.0}, OutType{-8.0, -5.0, -3.0, -2.0, -1.0}, "MultipleNegative"),
    std::make_tuple(InType{3.0, -1.0, 4.0, -5.0, 2.0, -3.0}, OutType{-5.0, -3.0, -1.0, 2.0, 3.0, 4.0}, "MixedSigns"),
    std::make_tuple(InType{0.0, -0.0, 1.0, -1.0}, SortedOutput({0.0, -0.0, 1.0, -1.0}), "WithZeros"),
    std::make_tuple(InType{-5.0, 0.0, 5.0}, OutType{-5.0, 0.0, 5.0}, "NegZeroPos"),
    std::make_tuple(InType{3.14, 2.72, 1.41, 1.62}, OutType{1.41, 1.62, 2.72, 3.14}, "DecimalNumbers"),
    std::make_tuple(InType{1e10, 1e5, 1e15, 1e1, 1e8}, OutType{1e1, 1e5, 1e8, 1e10, 1e15}, "LargePositive"),
    std::make_tuple(InType{-1e10, -1e5, -1e15, -1e1, -1e8}, OutType{-1e15, -1e10, -1e8, -1e5, -1e1}, "LargeNegative"),
    std::make_tuple(InType{1e-10, 1e-5, 1e-15, 1e-1, 1e-8}, OutType{1e-15, 1e-10, 1e-8, 1e-5, 1e-1}, "SmallPositive"),
    std::make_tuple(InType{-1e-10, -1e-5, -1e-15, -1e-1, -1e-8}, OutType{-1e-1, -1e-5, -1e-8, -1e-10, -1e-15},
                    "SmallNegative"),
    std::make_tuple(InType{5.0, 2.0, 5.0, 2.0, 5.0}, OutType{2.0, 2.0, 5.0, 5.0, 5.0}, "Duplicates"),
    std::make_tuple(InType{-3.0, -3.0, -3.0}, OutType{-3.0, -3.0, -3.0}, "AllSameNegative"),
    std::make_tuple(InType{1.0, 2.0, 3.0, 4.0, 5.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0}, "AlreadySorted"),
    std::make_tuple(InType{5.0, 4.0, 3.0, 2.0, 1.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0}, "ReverseSorted"),
    std::make_tuple(InType{8.0, 4.0, 2.0, 6.0, 1.0, 5.0, 7.0, 3.0}, OutType{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    "EightElements"),
    std::make_tuple(InType{4.0, 2.0, 3.0, 1.0}, OutType{1.0, 2.0, 3.0, 4.0}, "FourElements"),
    std::make_tuple(InType{-100.0, 50.0, -25.0, 75.0, 0.0, -50.0, 25.0, -75.0, 100.0},
                    OutType{-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0}, "NineElementsMixed"),
    std::make_tuple(InType{1.0, -1.0, 2.0, -2.0, 3.0, -3.0}, OutType{-3.0, -2.0, -1.0, 1.0, 2.0, 3.0},
                    "AlternatingSigns")};

TEST_P(EgashinKRunFuncTestsRadixBatcherSort, Sorting) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EgashinKRadixBatcherSortSEQ, InType>(kTestParam, PPC_SETTINGS_egashin_k_radix_batcher_sort),
    ppc::util::AddFuncTask<EgashinKRadixBatcherSortMPI, InType>(kTestParam, PPC_SETTINGS_egashin_k_radix_batcher_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    EgashinKRunFuncTestsRadixBatcherSort::PrintFuncTestName<EgashinKRunFuncTestsRadixBatcherSort>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(SortingTests, EgashinKRunFuncTestsRadixBatcherSort, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace egashin_k_radix_batcher_sort
