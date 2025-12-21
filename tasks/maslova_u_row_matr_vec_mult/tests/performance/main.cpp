#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"
#include "maslova_u_row_matr_vec_mult/mpi/include/ops_mpi.hpp"
#include "maslova_u_row_matr_vec_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_row_matr_vec_mult {

class MaslovaUPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_{};

 protected:
  void SetUp() override {
    const size_t rows = 5000;
    const size_t cols = 5000;

    input_data_.first.rows = rows;
    input_data_.first.cols = cols;
    input_data_.first.data.assign(rows * cols, 1.0);
    input_data_.second.assign(cols, 1.0);

    expected_output_.assign(rows, static_cast<double>(cols));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }
};

TEST_P(MaslovaUPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MaslovaURowMatrVecMultMPI, MaslovaURowMatrVecMultSEQ>(
        PPC_SETTINGS_maslova_u_row_matr_vec_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaslovaUPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaslovaUPerfTests, kGtestValues, kPerfTestName);

}  // namespace maslova_u_row_matr_vec_mult
