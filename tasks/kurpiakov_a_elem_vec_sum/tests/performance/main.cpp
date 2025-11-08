#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"
#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_elem_vec_sum {
class KurpiakovAElemVecSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{0, {}};
  OutType expected_data_;

  void SetUp() override {
    std::string input_data_source = ppc::util::GetAbsoluteTaskPath(PPC_ID_kurpiakov_a_elem_vec_sum, "test10_large.txt");

    std::ifstream file(input_data_source);
    int size = 0;
    double expected = NAN;
    std::vector<double> input;
    file >> size;
    file >> expected;
    double num = 0.0;
    while (file >> num) {
      input.push_back(num);
    }
    file.close();

    input_data_ = InType(size, input);
    expected_data_ = static_cast<OutType>(expected);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (std::abs(output_data - expected_data_) <= kEps);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KurpiakovAElemVecSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAElemVecSumMPI, KurpiakovAElemVecSumSEQ>(
    PPC_SETTINGS_kurpiakov_a_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovAElemVecSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KurpiakovAVecPerf, KurpiakovAElemVecSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_elem_vec_sum
