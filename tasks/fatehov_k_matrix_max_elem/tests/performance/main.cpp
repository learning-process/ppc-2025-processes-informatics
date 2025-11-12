#include <gtest/gtest.h>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_max_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKRunPerfTestsMatrixMaxElem : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType excepted_result_;
  void SetUp() override {
    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_fatehov_k_matrix_max_elem, "matrix.txt");
    std::ifstream file(file_path);
    size_t rows;
    size_t cols;
    double max_val;
    std::vector<double> matrix;

    file >> rows >> cols >> max_val;

    matrix.reserve(rows * cols);
    double value;
    for (size_t i = 0; i < rows * cols; ++i) {
      file >> value;
      matrix.push_back(value);
    }
    input_data_ = std::make_tuple(rows, cols, matrix);
    excepted_result_ = max_val;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return excepted_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FatehovKRunPerfTestsMatrixMaxElem, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FatehovKMatrixMaxElemMPI, FatehovKMatrixMaxElemSEQ>(
    PPC_SETTINGS_fatehov_k_matrix_max_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FatehovKRunPerfTestsMatrixMaxElem::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTest, FatehovKRunPerfTestsMatrixMaxElem, kGtestValues, kPerfTestName);

}  // namespace fatehov_k_matrix_max_elem
