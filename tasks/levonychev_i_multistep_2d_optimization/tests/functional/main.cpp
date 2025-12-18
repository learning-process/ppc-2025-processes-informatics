#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/mpi/include/ops_mpi.hpp"
#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_multistep_2d_optimization {

inline double ParaboloidFunction(double x, double y, double x0, double y0) {
  double dx = x - x0;
  double dy = y - y0;
  return dx * dx + dy * dy;
}

inline double RosenbrockFunction(double x, double y) {
  double term1 = 1.0 - x;
  double term2 = y - x * x;
  return term1 * term1 + 100.0 * term2 * term2;
}

inline double RastriginFunction(double x, double y) {
  const double pi = 3.14159265358979323846;
  const double A = 10.0;
  const double n = 2.0;

  double term1 = x * x - A * std::cos(2.0 * pi * x);
  double term2 = y * y - A * std::cos(2.0 * pi * y);

  return A * n + term1 + term2;
}

class LevonychevIMultistep2dOptimizationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    if (test_id == 1) {
      const double x_min = 0.0;
      const double x_max = 5.0;
      const double y_min = 0.0;
      const double y_max = 5.0;

      double x0 = 2.5;
      double y0 = 2.5;

      auto func = [x0, y0](double x, double y) -> double { return ParaboloidFunction(x, y, x0, y0); };

      input_data_.func = func;
      input_data_.x_min = x_min;
      input_data_.x_max = x_max;
      input_data_.y_min = y_min;
      input_data_.y_max = y_max;
      input_data_.num_steps = 3;
      input_data_.grid_size_step1 = 100;
      input_data_.candidates_per_step = 4;
      input_data_.use_local_optimization = true;

      expected_x_min_ = x0;
      expected_y_min_ = y0;
      expected_value_ = 0.0;

    } else if (test_id == 2) {
      const double x_min = 0.0;
      const double x_max = 2.0;
      const double y_min = 0.0;
      const double y_max = 2.0;

      auto func = [](double x, double y) -> double { return RosenbrockFunction(x, y); };

      input_data_.func = func;
      input_data_.x_min = x_min;
      input_data_.x_max = x_max;
      input_data_.y_min = y_min;
      input_data_.y_max = y_max;
      input_data_.num_steps = 5;
      input_data_.grid_size_step1 = 100;
      input_data_.candidates_per_step = 10;
      input_data_.use_local_optimization = true;

      expected_x_min_ = 1.0;
      expected_y_min_ = 1.0;
      expected_value_ = 0.0;

    } else if (test_id == 3) {
      const double x_min = -2.0;
      const double x_max = 2.0;
      const double y_min = -2.0;
      const double y_max = 2.0;

      auto func = [](double x, double y) -> double { return RastriginFunction(x, y); };

      input_data_.func = func;
      input_data_.x_min = x_min;
      input_data_.x_max = x_max;
      input_data_.y_min = y_min;
      input_data_.y_max = y_max;
      input_data_.num_steps = 5;
      input_data_.grid_size_step1 = 100;
      input_data_.candidates_per_step = 10;
      input_data_.use_local_optimization = true;

      expected_x_min_ = 0.0;
      expected_y_min_ = 0.0;
      expected_value_ = 0.0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.iterations <= 0) {
      return false;
    }

    if (output_data.x_min < input_data_.x_min || output_data.x_min > input_data_.x_max ||
        output_data.y_min < input_data_.y_min || output_data.y_min > input_data_.y_max) {
      return false;
    }

    if (!std::isfinite(output_data.value)) {
      return false;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    double tolerance_coord = 0.1;
    double tolerance_value = 0.01;

    if (test_id == 2) {
      tolerance_coord = 2;
      tolerance_value = 1;
    } else if (test_id == 3) {
      tolerance_coord = 3;
      tolerance_value = 5;
    }
    std::cout << test_id << " x_min: " << output_data.x_min << " y_min: " << output_data.y_min
              << " value: " << output_data.value << std::endl;
    bool x_ok = std::abs(output_data.x_min - expected_x_min_) < tolerance_coord;
    bool y_ok = std::abs(output_data.y_min - expected_y_min_) < tolerance_coord;
    bool value_ok = std::abs(output_data.value - expected_value_) < tolerance_value;

    return x_ok && y_ok && value_ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_x_min_ = 0.0;
  double expected_y_min_ = 0.0;
  double expected_value_ = 0.0;
};

namespace {

TEST_P(LevonychevIMultistep2dOptimizationFuncTests, Multistep2dOptimization) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "1"), std::make_tuple(2, "2"), std::make_tuple(3, "3")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LevonychevIMultistep2dOptimizationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_multistep_2d_optimization),
                                           ppc::util::AddFuncTask<LevonychevIMultistep2dOptimizationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_multistep_2d_optimization));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LevonychevIMultistep2dOptimizationFuncTests::PrintFuncTestName<LevonychevIMultistep2dOptimizationFuncTests>;

INSTANTIATE_TEST_SUITE_P(Multistep2dOptimizationTests, LevonychevIMultistep2dOptimizationFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace levonychev_i_multistep_2d_optimization
