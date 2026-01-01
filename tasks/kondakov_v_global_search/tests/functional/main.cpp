#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "kondakov_v_global_search/common/include/common.hpp"
#include "kondakov_v_global_search/mpi/include/ops_mpi.hpp"
#include "kondakov_v_global_search/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kondakov_v_global_search {

class KondakovVGlobalSearchRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto &params_tuple = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_params_ = std::get<0>(params_tuple);
    description_ = std::get<1>(params_tuple);
  }

  InType GetTestInputData() override {
    return test_params_;
  }

  bool CheckTestOutputData(OutType &output) override {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    const auto &p = test_params_;

    auto func = [p](double x) -> double {
      switch (p.func_type) {
        case FunctionType::kQuadratic:
          return (x - p.func_param) * (x - p.func_param);
        case FunctionType::kSine:
          return std::sin(x) + (0.1 * x);
        case FunctionType::kAbs:
          return std::abs(x);
        default:
          return 0.0;
      }
    };

    double recomputed = func(output.argmin);
    if (!std::isfinite(recomputed) || std::abs(recomputed - output.value) > 1e-6) {
      return false;
    }

    if (output.argmin < p.left - 1e-9 || output.argmin > p.right + 1e-9) {
      return false;
    }

    return output.iterations > 0 && output.iterations <= p.max_iterations;
  }

 private:
  Params test_params_;
  std::string description_;
};

namespace {

TEST_P(KondakovVGlobalSearchRunFuncTests, GlobalSearchCorrectness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParams = {{
    {Params{.func_type = FunctionType::kQuadratic,
            .func_param = 2.5,
            .left = 0.0,
            .right = 5.0,
            .accuracy = 1e-3,
            .reliability = 2.0,
            .max_iterations = 1000},
     "Quadratic_min_2_5"},
    {Params{.func_type = FunctionType::kSine,
            .func_param = 0.0,
            .left = -10.0,
            .right = 10.0,
            .accuracy = 1e-2,
            .reliability = 2.0,
            .max_iterations = 2000},
     "Sine_func_neg10_to_10"},
    {Params{.func_type = FunctionType::kAbs,
            .func_param = 0.0,
            .left = -3.0,
            .right = 3.0,
            .accuracy = 1e-3,
            .reliability = 1.5,
            .max_iterations = 1000},
     "Absolute_value_minimum"},
    {Params{.func_type = FunctionType::kQuadratic,
            .func_param = -1.0,
            .left = -5.0,
            .right = 0.0,
            .accuracy = 1e-3,
            .reliability = 2.5,
            .max_iterations = 1000},
     "Quadratic_minimum_neg1"},
}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KondakovVGlobalSearchMPI, InType>(kTestParams, PPC_SETTINGS_kondakov_v_global_search),
    ppc::util::AddFuncTask<KondakovVGlobalSearchSEQ, InType>(kTestParams, PPC_SETTINGS_kondakov_v_global_search));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KondakovVGlobalSearchRunFuncTests::PrintFuncTestName<KondakovVGlobalSearchRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(KondakovVGlobalSearchTests, KondakovVGlobalSearchRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace kondakov_v_global_search
