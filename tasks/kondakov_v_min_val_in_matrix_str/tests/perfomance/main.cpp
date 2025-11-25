#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"
#include "kondakov_v_min_val_in_matrix_str/mpi/include/ops_mpi.hpp"
#include "kondakov_v_min_val_in_matrix_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kondakov_v_min_val_in_matrix_str {

class KondakovVMinValMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const size_t N = 1000;
    input_data_.resize(N, std::vector<int>(N));

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N - 1; ++j) {
        input_data_[i][j] = static_cast<int>(j + 1);
      }
      input_data_[i][N - 1] = -static_cast<int>(i + 1);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != -static_cast<int>(i + 1)) {
        return false;
      }
    }
    return true;
  }
  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KondakovVMinValMatrixPerfTests, FindMinInRowsPerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KondakovVMinValMatrixMPI, KondakovVMinValMatrixSEQ>(
    PPC_SETTINGS_kondakov_v_min_val_in_matrix_str);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondakovVMinValMatrixPerfTests::CustomPerfTestName;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(MinValMatrixPerf, KondakovVMinValMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace kondakov_v_min_val_in_matrix_str
