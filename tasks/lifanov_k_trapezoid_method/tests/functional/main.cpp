#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"
#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lifanov_k_trapezoid_method {

using TestParams = std::tuple<int, int, std::string>;  
// nx, ny, name

class LifanovKTrapezoidMethodFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestParams> {
 public:
  static std::string PrintTestParam(const TestParams& param) {
    return std::to_string(std::get<0>(param)) + "x" +
           std::to_string(std::get<1>(param)) + "_" +
           std::get<2>(param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(
        ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int nx = std::get<0>(params);
    int ny = std::get<1>(params);

    // Вход: a, b, c, d, nx, ny
    input_data_ = {
        0.0, 1.0,
        0.0, 1.0,
        static_cast<double>(nx),
        static_cast<double>(ny)};
  }

  bool CheckTestOutputData(OutType& output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    const double exact = 2.0 / 3.0;
    const double eps = 1e-1;

    return std::abs(output_data - exact) < eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LifanovKTrapezoidMethodFuncTests, Trapezoid2D) {
  ExecuteTest(GetParam());
}

const std::array<TestParams, 3> kTestParams = {
    std::make_tuple(50, 50, "50x50"),
    std::make_tuple(100, 100, "100x100"),
    std::make_tuple(300, 300, "300x300")};

const auto kTestTasksList =
    std::tuple_cat(
        ppc::util::AddFuncTask<LifanovKTrapezoidMethodMPI, InType>(
            kTestParams, PPC_SETTINGS_lifanov_k_trapezoid_method),
        ppc::util::AddFuncTask<LifanovKTrapezoidMethodSEQ, InType>(
            kTestParams, PPC_SETTINGS_lifanov_k_trapezoid_method));

const auto kGtestValues =
    ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName =
    LifanovKTrapezoidMethodFuncTests::PrintFuncTestName<
        LifanovKTrapezoidMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(
    LifanovKTrapezoidMethodTests,
    LifanovKTrapezoidMethodFuncTests,
    kGtestValues,
    kTestName);

}  // namespace

}  // namespace lifanov_k_trapezoid_method
