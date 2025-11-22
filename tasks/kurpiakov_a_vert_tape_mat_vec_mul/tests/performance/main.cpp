#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

class KurpiakovARunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType expected_data_{};

  void SetUp() override {
    std::vector<long long> matrix(5000 * 5000);
    std::vector<long long> vector(5000);    
    
    for (int i = 0; i < 5000; ++i) {
      matrix[i * 5000 + i] = i;
      expected_data_.push_back(matrix[i * 5000 + i]);
    }
    
    for (int i = 0; i < 5000; ++i) {
      vector[i] = 1;
    }
    
    input_data_ = std::make_tuple(5000, matrix, vector);
  
  }

  bool CheckTestOutputData(OutType &output_data) final {


    for (int i = 0; i < static_cast<int>(expected_data_.size()); ++i) {
      if (expected_data_[i] != output_data[i]) {
        return false;
      }
    }

    
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KurpiakovARunPerfTestProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAVretTapeMulMPI, KurpiakovAVretTapeMulSEQ>(
    PPC_SETTINGS_kurpiakov_a_vert_tape_mat_vec_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovARunPerfTestProcesses2::CustomPerfTestName;

//NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(MatVecMulTestsPerf, KurpiakovARunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
