#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

class KiselevIPerfTests2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t num = 6000;
    const std::size_t band = 6;

    std::vector<std::vector<double>> aVector(num, std::vector<double>(num, 0.0));

    for (std::size_t index = 0; index < num; ++index) {
      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      double sum_abs = 0.0;
      for (std::size_t jIndex = j_begin; jIndex < j_end; ++jIndex) {
        if (jIndex == index) {
          continue;
        }

        const double val = (jIndex < index) ? 0.5 : 1.5;
        aVector[index][jIndex] = val;
        sum_abs += std::abs(val);
      }

      aVector[index][index] = sum_abs + 8.0 + static_cast<double>(index % 3);
    }

    std::vector<double> x_true(num);
    for (std::size_t index = 0; index < num; ++index) {
      x_true[index] = 1.0 + static_cast<double>((index * 3) % 11);
    }

    std::vector<double> bVector(num, 0.0);
    for (std::size_t index = 0; index < num; ++index) {
      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      double s = 0.0;
      for (std::size_t jIndex = j_begin; jIndex < j_end; ++jIndex) {
        s += aVector[index][jIndex] * x_true[jIndex];
      }
      bVector[index] = s;
    }

    input_data_ = std::make_tuple(std::move(aVector), std::move(bVector), band);
  }

  bool CheckTestOutputData(OutType &output) final {
    const auto &aVector = std::get<0>(input_data_);
    const auto &bVector = std::get<1>(input_data_);
    const std::size_t band = std::get<2>(input_data_);

    const std::size_t num = aVector.size();
    if (output.size() != num) {
      return false;
    }

    double max_residual = 0.0;

    for (std::size_t index = 0; index < num; ++index) {
      double s = 0.0;

      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      for (std::size_t jIndex = j_begin; jIndex < j_end; ++jIndex) {
        s += aVector[index][jIndex] * output[jIndex];
      }

      max_residual = std::max(max_residual, std::abs(s - bVector[index]));
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
