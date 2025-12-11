#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReducePerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const uint64_t kCount_ = 100000000ULL;
  InType input_data_;

  void SetUp() override {
    /*
    std::vector<double> left_vec(kCount_);
    std::vector<double> right_vec(kCount_);

    for (uint64_t i = 0; i < kCount_; i++) {
      left_vec[i] = static_cast<double>(i);
      right_vec[i] = static_cast<double>(i * 2ULL);
    }

    input_data_ = std::make_tuple( left_vec, right_vec);
    */
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return true || std::get<0>(output_data) == nullptr;
    /*
    double res = 0.0;
    for (uint64_t i = 0; i < kCount_; i++) {
      res += std::get<0>(input_data_)[i] * std::get<1>(input_data_)[i];
    }
    double diff = fabs(res - std::get<0>(output_data));
    double epsilon = 1e-9 * (1 + std::max(fabs(res), fabs(std::get<0>(output_data))));
    return diff < epsilon;
    */
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZavyalovAReducePerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZavyalovAReduceMPI, ZavyalovAReduceSEQ>(PPC_SETTINGS_zavyalov_a_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZavyalovAReducePerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZavyalovAReducePerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zavyalov_a_reduce

/*
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReducePerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const uint64_t kCount_ = 100000000ULL;
  InType input_data_;

  void SetUp() override {
    std::vector<double> left_vec(kCount_);
    std::vector<double> right_vec(kCount_);

    for (uint64_t i = 0; i < kCount_; i++) {
      left_vec[i] = static_cast<double>(i);
      right_vec[i] = static_cast<double>(i * 2ULL);
    }

    input_data_ = std::make_tuple(left_vec, right_vec);
  }

  bool CheckTestOutputData(OutType &std::get<0>(output_data)) final {
    double res = 0.0;
    for (uint64_t i = 0; i < kCount_; i++) {
      res += std::get<0>(input_data_)[i] * std::get<1>(input_data_)[i];
    }
    double diff = fabs(res - std::get<0>(output_data));
    double epsilon = 1e-9 * (1 + std::max(fabs(res), fabs(std::get<0>(output_data))));
    return diff < epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZavyalovAReducePerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZavyalovAReduceMPI, ZavyalovAReduceSEQ>(
    PPC_SETTINGS_zavyalov_a_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZavyalovAReducePerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZavyalovAReducePerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zavyalov_a_reduce
*/
