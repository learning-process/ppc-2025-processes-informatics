#include <gtest/gtest.h>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_dot_product_of_vectors {

class SpichekDDotProductOfVectorsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const InType kVectorSize_ = 1000;  // Больший размер для производительности
  InType input_data_{};

  void SetUp() override {
    input_data_ = kVectorSize_;
  }

bool CheckTestOutputData(OutType &output_data) final {
    InType n = input_data_;
    InType expected_result = 0;
    
    for (InType i = 1; i <= n; i++) {
        expected_result += i * i;
    }
    
    return (output_data == expected_result);
}

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SpichekDDotProductOfVectorsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SpichekDDotProductOfVectorsMPI, SpichekDDotProductOfVectorsSEQ>(PPC_SETTINGS_spichek_d_dot_product_of_vectors);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SpichekDDotProductOfVectorsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SpichekDDotProductOfVectorsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace spichek_d_dot_product_of_vectors