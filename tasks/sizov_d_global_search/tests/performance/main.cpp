#include <gtest/gtest.h>

#include <cmath>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sizov_d_global_search {

namespace {

constexpr double kLeft = -5.0;
constexpr double kRight = 5.0;
constexpr double kAccuracy = 1e-4;
constexpr double kReliability = 3.0;
constexpr int kMaxIterations = 150000;

InType MakePerfProblem() {
  InType p{};
  p.left = kLeft;
  p.right = kRight;
  p.accuracy = kAccuracy;
  p.reliability = kReliability;
  p.max_iterations = kMaxIterations;

  p.func = [](double x) {
    double v = 0.002 * x * x;
    v += 5.0 * std::sin(30.0 * x);
    v += std::sin(200.0 * std::sin(50.0 * x));
    v += 0.1 * std::cos(300.0 * x);
    return v;
  };
  return p;
}

class SizovDGlobalSearchPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_{};

  bool CheckTestOutputData(OutType &out) final {
    if (!out.converged) {
      return false;
    }
    if (!std::isfinite(out.value)) {
      return false;
    }
    if (out.argmin < input_data_.left - 1e-6 || out.argmin > input_data_.right + 1e-6) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 protected:
  void SetUp() override {
    input_data_ = MakePerfProblem();
  }
};

TEST_P(SizovDGlobalSearchPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SizovDGlobalSearchMPI, SizovDGlobalSearchSEQ>(
    PPC_SETTINGS_sizov_d_global_search);

}  // namespace

INSTANTIATE_TEST_SUITE_P(RunModeTests, SizovDGlobalSearchPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         SizovDGlobalSearchPerfTests::CustomPerfTestName);

}  // namespace sizov_d_global_search
