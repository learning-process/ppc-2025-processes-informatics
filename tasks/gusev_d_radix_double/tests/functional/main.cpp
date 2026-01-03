#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "gusev_d_radix_double/common/include/common.hpp"
#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_radix_double {

class GusevDRadixDoubleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    std::string test_name = std::get<1>(params);

    input_data_ = std::vector<double>(count);
    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)

    if (test_name.find("Positive") != std::string::npos) {
      std::uniform_real_distribution<> dis(0.1, 1000.0);
      for (int i = 0; i < count; ++i) {
        input_data_[i] = dis(gen);
      }
    } else if (test_name.find("Negative") != std::string::npos) {
      std::uniform_real_distribution<> dis(-1000.0, -0.1);
      for (int i = 0; i < count; ++i) {
        input_data_[i] = dis(gen);
      }
    } else if (test_name.find("Zero") != std::string::npos) {
      std::fill(input_data_.begin(), input_data_.end(), 0.0);  // NOLINT(modernize-use-ranges)
    } else {
      std::uniform_real_distribution<> dis(-1000.0, 1000.0);
      for (int i = 0; i < count; ++i) {
        input_data_[i] = dis(gen);
      }
    }

    if (test_name.find("Sorted") != std::string::npos && test_name.find("Reverse") == std::string::npos) {
      std::sort(input_data_.begin(), input_data_.end());  // NOLINT(modernize-use-ranges)
    } else if (test_name.find("Reverse") != std::string::npos) {
      std::sort(input_data_.begin(), input_data_.end(), std::greater<>());  // NOLINT(modernize-use-ranges)
    }

    ref_output_data_ = input_data_;
    std::sort(ref_output_data_.begin(), ref_output_data_.end());  // NOLINT(modernize-use-ranges)
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return output_data == ref_output_data_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_data_;
};

namespace {

TEST_P(GusevDRadixDoubleFuncTests, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParams = {
    std::make_tuple(10, "SmallVector"),     std::make_tuple(100, "MediumVector"),
    std::make_tuple(500, "LargeVector"),    std::make_tuple(0, "EmptyVector"),
    std::make_tuple(1, "SingleElement"),

    std::make_tuple(100, "PositiveVector"), std::make_tuple(100, "NegativeVector"),
    std::make_tuple(100, "SortedVector"),   std::make_tuple(100, "ReverseSortedVector"),
    std::make_tuple(50, "ZeroVector"),

    std::make_tuple(123, "OddSizeVector"),

    std::make_tuple(1000, "Size_1000"),     std::make_tuple(2048, "Size_2048_Pow2"),
    std::make_tuple(2500, "Size_2500"),     std::make_tuple(5000, "Size_5000_MaxFunc")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GusevDRadixDoubleSEQ, InType>(kTestParams, PPC_SETTINGS_gusev_d_radix_double),
    ppc::util::AddFuncTask<GusevDRadixDoubleMPI, InType>(kTestParams, PPC_SETTINGS_gusev_d_radix_double));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevDRadixDoubleFuncTests::PrintFuncTestName<GusevDRadixDoubleFuncTests>;

INSTANTIATE_TEST_SUITE_P(RadixSortDoubleTests, GusevDRadixDoubleFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gusev_d_radix_double
