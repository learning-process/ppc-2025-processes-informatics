#include <gtest/gtest.h>

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "kichanova_k_count_letters_in_str/mpi/include/ops_mpi.hpp"
#include "kichanova_k_count_letters_in_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kichanova_k_count_letters_in_str {

class KichanovaKCountLettersInStrPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  int expected_output_{};

   void SetUp() override {
    const size_t str_size = 5000000; 
    
    std::string generated_string;
    generated_string.reserve(str_size);
    
    for (size_t i = 0; i < str_size; ++i) {
        if (i % 3 == 0) {
            generated_string += (i % 2 == 0) ? 'a' + (i % 26) : 'A' + (i % 26);
            expected_output_++;
        } else {
            generated_string += (i % 2 == 0) ? '0' + (i % 10) : '!' + (i % 15);
        }
    }
    
    input_data_ = generated_string;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KichanovaKCountLettersInStrPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KichanovaKCountLettersInStrMPI, KichanovaKCountLettersInStrSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KichanovaKCountLettersInStrPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KichanovaKCountLettersInStrPerfTest, kGtestValues, kPerfTestName);


}  // namespace kichanova_k_count_letters_in_str
