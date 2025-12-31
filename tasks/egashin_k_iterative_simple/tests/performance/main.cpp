#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "egashin_k_iterative_simple/mpi/include/ops_mpi.hpp"
#include "egashin_k_iterative_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_iterative_simple {

class EgashinKRunPerfTestIterativeSimple : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 5000;
  InType input_data_{};

  void SetUp() override {
    std::vector<std::vector<double>> matrix(kCount_, std::vector<double>(kCount_, 0.0));
    std::vector<double> x0(kCount_, 0.0);

    for (int i = 0; i < kCount_; ++i) {
      matrix[i][i] = 4.0;
      if (i > 0) {
        matrix[i][i - 1] = -1.0;
      }
      if (i < kCount_ - 1) {
        matrix[i][i + 1] = -1.0;
      }
    }

    std::vector<double> b(kCount_, 2.0);
    if (kCount_ > 0) {
      b[0] = 3.0;
      b[static_cast<std::size_t>(kCount_ - 1)] = 3.0;
    }

    input_data_.A = matrix;
    input_data_.b = b;
    input_data_.x0 = x0;
    input_data_.tolerance = 1e-6;
    input_data_.max_iterations = 10000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    OutType expected_data(kCount_, 1.0);
    if (output_data.size() != expected_data.size()) {
      return false;
    }

    double tolerance = input_data_.tolerance * 100;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_data[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(EgashinKRunPerfTestIterativeSimple, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, EgashinKIterativeSimpleMPI, EgashinKIterativeSimpleSEQ>(
    PPC_SETTINGS_egashin_k_iterative_simple);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EgashinKRunPerfTestIterativeSimple::CustomPerfTestName;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(RunModeTests, EgashinKRunPerfTestIterativeSimple, kGtestValues, kPerfTestName);

}  // namespace egashin_k_iterative_simple
