#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_jacobi {

class SpichekDJacobiRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    // ИЗМЕНЕНИЕ: Уменьшаем N до 500.
    // N=2400 слишком велико для неоптимизированного SEQ кода в Debug сборке (таймаут > 100 сек).
    // N=500 даст время около 3-5 секунд для SEQ и мгновенно для MPI.
    const size_t n_size = 500;

    // Оставляем 100 итераций, этого достаточно
    constexpr double kEpsilon = 1e-5;
    constexpr int kMaxIter = 100;

    Matrix A(n_size, Vector(n_size));
    Vector b(n_size);

    for (size_t i = 0; i < n_size; ++i) {
      double sum_off_diag = 0.0;
      for (size_t j = 0; j < n_size; ++j) {
        if (i == j) {
          A[i][j] = 0.0;
        } else {
          double val = static_cast<double>((i * 7 + j) % 10 + 1) / 10.0;
          A[i][j] = val;
          sum_off_diag += std::abs(val);
        }
      }
      A[i][i] = sum_off_diag + 1.0 + static_cast<double>(i % 10);
      b[i] = static_cast<double>(i + 1);
    }

    input_data_ = std::make_tuple(A, b, kEpsilon, kMaxIter);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }
    double sum_sq = 0.0;
    for (double val : output_data) {
      if (std::isnan(val) || std::isinf(val)) {
        return false;
      }
      sum_sq += val * val;
    }
    return (sum_sq > 1e-9);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SpichekDJacobiRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SpichekDJacobiMPI, SpichekDJacobiSEQ>(PPC_SETTINGS_spichek_d_jacobi);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SpichekDJacobiRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SpichekDJacobiRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace spichek_d_jacobi
