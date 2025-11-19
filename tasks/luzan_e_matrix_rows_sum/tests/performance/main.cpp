#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_matrix_rows_sum {

class LuzanEMatrixRowsSumpERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t height_ = 10000;
  const size_t width_ = 10000;
  InType input_data_;

  void SetUp() override {
    std::tuple_element_t<0, InType> mat(height_ * width_);
    for (size_t elem = 0; elem < height_ * width_; elem++) {
      mat[elem] = static_cast<int>(elem) % 42000;
    }

    input_data_ = std::make_tuple(mat, height_, width_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> sum(height_);
    std::tuple_element_t<0, InType> mat = std::get<0>(input_data_);

    for (size_t row = 0; row < height_; row++) {
      for (size_t col = 0; col < width_; col++) {
        sum[row] += mat[(width_ * row) + col];
      }
    }

    return (output_data == sum);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanEMatrixRowsSumpERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuzanEMatrixRowsSumMPI, LuzanEMatrixRowsSumSEQ>(
    PPC_SETTINGS_luzan_e_matrix_rows_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanEMatrixRowsSumpERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanEMatrixRowsSumpERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_matrix_rows_sum
