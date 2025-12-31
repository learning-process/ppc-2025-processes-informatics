#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

class KiselevIPerfTests2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t n = 6000;
    const std::size_t band = 6;

    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j_begin = (i > band) ? (i - band) : 0;
      const std::size_t j_end = std::min(n, i + band + 1);

      double sum_abs = 0.0;
      for (std::size_t j = j_begin; j < j_end; ++j) {
        if (j == i) {
          continue;
        }

        const double val = (j < i) ? 0.5 : 1.5;
        a[i][j] = val;
        sum_abs += std::abs(val);
      }

      a[i][i] = sum_abs + 8.0 + static_cast<double>(i % 3);
    }

    std::vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i) {
      x_true[i] = 1.0 + static_cast<double>((i * 3) % 11);
    }

    std::vector<double> b(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j_begin = (i > band) ? (i - band) : 0;
      const std::size_t j_end = std::min(n, i + band + 1);

      double s = 0.0;
      for (std::size_t j = j_begin; j < j_end; ++j) {
        s += a[i][j] * x_true[j];
      }
      b[i] = s;
    }

    input_data_ = std::make_tuple(std::move(a), std::move(b), band);
  }

  bool CheckTestOutputData(OutType &output) final {
    const auto &a = std::get<0>(input_data_);
    const auto &b = std::get<1>(input_data_);
    const std::size_t band = std::get<2>(input_data_);

    const std::size_t n = a.size();
    if (output.size() != n) {
      return false;
    }

    double max_residual = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;

      const std::size_t j_begin = (i > band) ? (i - band) : 0;
      const std::size_t j_end = std::min(n, i + band + 1);

      for (std::size_t j = j_begin; j < j_end; ++j) {
        s += a[i][j] * output[j];
      }

      max_residual = std::max(max_residual, std::abs(s - b[i]));
    }

    return max_residual < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KiselevIPerfTests2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(
    PPC_SETTINGS_kiselev_i_gauss_method_horizontal_tape_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIPerfTests2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIPerfTests2, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
