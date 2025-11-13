#include <gtest/gtest.h>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_max_elem_matrix {

class PylaevaSMaxElemMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{0, {}};
  OutType expected_data_;

  void SetUp() override {
    std::string filename = ppc::util::GetAbsoluteTaskPath(PPC_ID_pylaeva_s_max_elem_matrix, "matrix_4096x4096.txt");

    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }
    
    size_t size = 0;
    std::vector<double> input;
    input.reserve(size);
    int max;

    file >> size;
    file >> max;

    int elem = 0;
    while (file >> elem) {
      input.push_back(elem);
    }

    input_data_ = std::make_tuple(size, input);
    expected_data_ = max;

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_data_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PylaevaSMaxElemMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PylaevaSMaxElemMatrixMPI, PylaevaSMaxElemMatrixSEQ>(PPC_SETTINGS_pylaeva_s_max_elem_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSMaxElemMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSMaxElemMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_max_elem_matrix
