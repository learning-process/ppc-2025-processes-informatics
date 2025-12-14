#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "krykov_e_simple_iteration/common/include/common.hpp"
#include "krykov_e_simple_iteration/mpi/include/ops_mpi.hpp"
#include "krykov_e_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krykov_e_simple_iteration {

class KrykovESimpleIterationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    constexpr size_t n = 200;

    std::vector<double> A(n * n, 0.0);
    std::vector<double> b(n, 0.0);
    expected_output_.assign(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
      double row_sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          A[i * n + j] = 1.0;
          row_sum += 1.0;
        }
      }
      A[i * n + i] = row_sum + 5.0;  // строгая доминанта
      b[i] = A[i * n + i] + row_sum;
    }

    input_data_ = std::make_tuple(n, A, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double eps = 1e-4;
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  KrykovESimpleIterationPerfTests() = default;
};

TEST_P(KrykovESimpleIterationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KrykovESimpleIterationMPI, KrykovESimpleIterationSEQ>(
    PPC_SETTINGS_krykov_e_simple_iteration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrykovESimpleIterationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrykovESimpleIterationPerfTests, kGtestValues, kPerfTestName);

}  // namespace krykov_e_simple_iteration
