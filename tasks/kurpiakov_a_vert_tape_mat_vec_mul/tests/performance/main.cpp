#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

class KurpiakovAVertTapeMatVecMulPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType expected_data_{};

  void SetUp() override {
    const int64_t size = 8500;

    std::vector<int64_t> matrix(static_cast<size_t>(size * size));
    std::vector<int64_t> vector(static_cast<size_t>(size));

    for (int64_t i = 0; i < size; ++i) {
      for (int64_t j = 0; j < size; ++j) {
        matrix[static_cast<size_t>(i * size + j)] = (i == j) ? (i + 1) : 0;
      }
      expected_data_.push_back(i + 1);
    }

    for (int64_t i = 0; i < size; ++i) {
      vector[i] = 1LL;
    }

    input_data_ = std::make_tuple(size, matrix, vector);
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

TEST_P(KurpiakovAVertTapeMatVecMulPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAVretTapeMulMPI, KurpiakovAVretTapeMulSEQ>(
    PPC_SETTINGS_kurpiakov_a_vert_tape_mat_vec_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovAVertTapeMatVecMulPerfTests::CustomPerfTestName;
// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(RunModeTests, KurpiakovAVertTapeMatVecMulPerfTests, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
