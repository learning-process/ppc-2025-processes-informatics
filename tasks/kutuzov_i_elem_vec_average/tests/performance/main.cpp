#include <gtest/gtest.h>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_elem_vec_average/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutuzov_i_elem_vec_average {

class KutuzovIElemVecAveragePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_data_{};

  void SetUp() override {

    input_data_ = std::vector<double>(kCount_);

    for (int i = 0; i < kCount_; i++)
      input_data_[i] = i * i - (double)kCount_ / 2;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double average = 0.0;
    for (size_t i = 0; i < input_data_.size(); i++)
      average += input_data_[i];

    average /= input_data_.size();

    return abs(output_data - average) < 0.0001;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KutuzovIElemVecAveragePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KutuzovIElemVecAverageMPI, KutuzovIElemVecAverageSEQ>(PPC_SETTINGS_kutuzov_i_elem_vec_average);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KutuzovIElemVecAveragePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KutuzovIElemVecAveragePerfTests, kGtestValues, kPerfTestName);

}  // namespace kutuzov_i_elem_vec_average
