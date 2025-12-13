#include <gtest/gtest.h>
#include <mpi.h>

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

 protected:
  void SetUp() override {
    // УМЕНЬШЕНО до 128.
    // 500 слишком много для SEQ версии в рамках pipeline теста на некоторых машинах.
    const size_t n_size = 128;

    constexpr double kEpsilon = 1e-5;
    constexpr int kMaxIter = 500;

    Matrix A(n_size, Vector(n_size));
    Vector b(n_size);

    for (size_t i = 0; i < n_size; ++i) {
      double sum_off_diag = 0.0;
      for (size_t j = 0; j < n_size; ++j) {
        if (i == j) {
          A[i][j] = 0.0;
        } else {
          // Простая генерация, чтобы не тратить время setup
          double val = 0.1;
          A[i][j] = val;
          sum_off_diag += std::abs(val);
        }
      }
      // Гарантируем диагональное преобладание
      A[i][i] = sum_off_diag + 1.0;
      b[i] = 1.0;
    }

    input_data_ = std::make_tuple(A, b, kEpsilon, kMaxIter);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    if (output_data.empty()) {
      return false;
    }

    // Простая проверка, что данные не NaN/Inf и не нулевые
    double sum_sq = 0.0;
    for (double val : output_data) {
      if (std::isnan(val) || std::isinf(val)) {
        return false;
      }
      sum_sq += val * val;
    }
    std::cout << sum_sq << std::endl;
    return sum_sq > 1e-9;
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
