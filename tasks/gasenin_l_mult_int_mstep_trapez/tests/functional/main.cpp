#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"
#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLIntegralFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    double expected = GetExactIntegral(input_data_);

    double tolerance = (input_data_.n_steps <= 20) ? 1e-2 : 1e-3;

    return std::abs(output_data - expected) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(GaseninLIntegralFunctionalTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParams = {std::make_tuple(TaskData{20, 0, 0.0, 1.0, 0.0, 1.0}, "Func0_Grid20"),
                                             std::make_tuple(TaskData{50, 0, 0.0, 1.0, 0.0, 1.0}, "Func0_Grid50"),
                                             std::make_tuple(TaskData{100, 1, 0.0, 1.0, 0.0, 1.0}, "Func1_Grid100"),
                                             std::make_tuple(TaskData{200, 1, 0.0, 1.0, 0.0, 1.0}, "Func1_Grid200")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezMPI, InType>(
                                               kTestParams, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez),
                                           ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestNameFormatter = GaseninLIntegralFunctionalTests::PrintFuncTestName<GaseninLIntegralFunctionalTests>;

INSTANTIATE_TEST_SUITE_P(GaseninLFunctionalTestSuite, GaseninLIntegralFunctionalTests, kGtestValues,
                         kTestNameFormatter);

}  // namespace gasenin_l_mult_int_mstep_trapez
