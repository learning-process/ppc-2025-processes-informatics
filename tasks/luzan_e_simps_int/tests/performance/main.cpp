#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_simps_int/common/include/common.hpp"
#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"
#include "luzan_e_simps_int/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntpERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const double a_ = 1.0, b_ = 5.0;
  const double c_ = 1.0, d_ = 5.0;
  const int n_ = 10;
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_tuple(n_, std::make_tuple(a_, b_), std::make_tuple(c_, d_));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double a = 0.0, b = 0.0;
    double c = 0.0, d = 0.0;
    int n = 0;

    // getting data
    n = std::get<0>(input_data_);
    a = std::get<0>(std::get<1>(input_data_));
    b = std::get<1>(std::get<1>(input_data_));
    c = std::get<0>(std::get<2>(input_data_));
    d = std::get<1>(std::get<2>(input_data_));

    double hx = (b - a) / n;
    double hy = (d - c) / n;

    double sum = 0;
    double wx = 1.0, wy = 1.0;
    double x = 0.0, y = 0.0;

    for (int i = 0; i <= n; i++) {
      x = a + hx * i;
      wx = 1.0;
      if (i != 0 && i != n) {
        if (i % 2 == 1) {
          wx = 4.0;
        } else {
          wx = 2.0;
        }
      }

      for (int j = 0; j <= n; j++) {
        wy = 1.0;
        y = c + hy * j;

        if (j != 0 && j != n) {
          if (j % 2 == 1) {
            wy = 4.0;
          } else {
            wy = 2.0;
          }
        }
        sum += wy * wx * f(x, y);
      }
    }
    sum = sum * hx * hy / (3 * 3);

    return (abs(output_data - sum) < EPSILON);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanESimpsIntpERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuzanESimpsIntMPI, LuzanESimpsIntSEQ>(PPC_SETTINGS_luzan_e_simps_int);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanESimpsIntpERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanESimpsIntpERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_simps_int
