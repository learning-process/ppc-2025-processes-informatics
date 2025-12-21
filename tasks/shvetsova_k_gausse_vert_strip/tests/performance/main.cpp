#include <gtest/gtest.h>

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Берём данные из функционального теста
    std::vector<std::vector<double>> matrix = {{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 5.0}};
    std::vector<double> b = {2.0, 8.0, 10.0};
    input_data_ = {matrix, b};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-6;
    const auto &matrix = input_data_.first;
    const auto &b = input_data_.second;
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[i][j] * output_data[j];
      }
      if (std::abs(sum - b[i]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

// Регистрация задач с seq solver
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKGaussVertStripSEQ>(PPC_SETTINGS_shvetsova_k_gausse_vert_strip);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKGaussVertStripPerfTest::CustomPerfTestName;

TEST_P(ShvetsovaKGaussVertStripPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKGaussVertStripPerfTest, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_gausse_vert_strip
