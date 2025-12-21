// #include <gtest/gtest.h>

// #include "guseva_a_jarvis/common/include/common.hpp"
// #include "guseva_a_jarvis/mpi/include/ops_mpi.hpp"
// #include "guseva_a_jarvis/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace guseva_a_jarvis {

// class GusevaAJarvisPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const int kCount_ = 100;
//   InType input_data_{};

//   void SetUp() override {
//     input_data_ = kCount_;
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     return input_data_ == output_data;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(GusevaAJarvisPerfTests, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, GusevaAJarvisMPI, GusevaAJarvisSEQ>(PPC_SETTINGS_guseva_a_jarvis);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = GusevaAJarvisPerfTests::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevaAJarvisPerfTests, kGtestValues, kPerfTestName);

// }  // namespace guseva_a_jarvis
