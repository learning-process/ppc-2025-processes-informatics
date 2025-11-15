#include <gtest/gtest.h>

#include <iostream>

#include "util/include/perf_test_util.hpp"
#include "zavyalov_a_scalar_product/common/include/common.hpp"
#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_scalar_product/seq/include/ops_seq.hpp"

namespace zavyalov_a_scalar_product {

class ZavyalovAScalarProductPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const unsigned long long kCount_ = 30000000;
  InType input_data_{};

  void SetUp() override {
    std::vector<double> leftVec(kCount_);
    std::vector<double> rightVec(kCount_);

    for (unsigned long long i = 0; i < kCount_; i++) {
      leftVec[i] = i;
      rightVec[i] = i * 2ull;
    }

    input_data_ = std::make_tuple(leftVec, rightVec);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double res = 0.0;
    for (unsigned long long i = 0; i < kCount_; i++) {
      res += std::get<0>(input_data_)[i] * std::get<1>(input_data_)[i];
    }
    double diff = fabs(res - output_data);
    double epsilon = 1e-9 * (1 + std::max(fabs(res), fabs(output_data)));
    return diff < epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZavyalovAScalarProductPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZavyalovAScalarProductMPI, ZavyalovAScalarProductSEQ>(
    PPC_SETTINGS_zavyalov_a_scalar_product);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZavyalovAScalarProductPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZavyalovAScalarProductPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zavyalov_a_scalar_product
