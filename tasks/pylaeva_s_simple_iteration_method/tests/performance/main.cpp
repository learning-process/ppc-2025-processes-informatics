#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_data_{};
  const size_t N = 300;

  void SetUp() override {
    std::vector<double> A(N * N, 1.0);
    std::vector<double> b(N, 0.0);
    expected_data_.assign(N, 1.0);

    for (size_t i = 0; i < N; ++i) {
      A[(i * N) + i] = N;
      b[i] = (2 * N) - 1.0;
    }

    input_data_ = std::make_tuple(N, A, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double EPS = 1e-6;
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_data_[i]) > EPS) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  PylaevaSSimpleIterationMethodPerfTests() = default;
};

TEST_P(PylaevaSSimpleIterationMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PylaevaSSimpleIterationMethodMPI, PylaevaSSimpleIterationMethodSEQ>(
        PPC_SETTINGS_pylaeva_s_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSSimpleIterationMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSSimpleIterationMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_simple_iteration_method
