#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"
#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lifanov_k_trapezoid_method {

class LifanovKTrapezoidMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::vector<int>(std::get<0>(params), 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    return output_data.size() == 1 && output_data[0] != 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LifanovKTrapezoidMethodFuncTests, MyTrapezoidMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(11, "11"), std::make_tuple(50, "50"),
                                            std::make_tuple(123, "123")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LifanovKTrapezoidMethodMPI, InType>(kTestParam, PPC_SETTINGS_lifanov_k_trapezoid_method),
                   ppc::util::AddFuncTask<LifanovKTrapezoidMethodSEQ, InType>(kTestParam, PPC_SETTINGS_lifanov_k_trapezoid_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LifanovKTrapezoidMethodFuncTests::PrintFuncTestName<LifanovKTrapezoidMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(MyTrapezoidMethodTests, LifanovKTrapezoidMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lifanov_k_trapezoid_method
