#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

using makovskiy_i_gauss_filter_vert::GaussFilterMPI;
using makovskiy_i_gauss_filter_vert::GaussFilterSEQ;
using makovskiy_i_gauss_filter_vert::InType;
using makovskiy_i_gauss_filter_vert::OutType;

using TestDS = std::pair<InType, OutType>;

class GaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestDS> {
 protected:
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  InType GetTestInputData() override {
    return std::get<2>(GetParam()).first;
  }

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  bool CheckTestOutputData(OutType &result) override {
    const auto &expected = std::get<2>(GetParam()).second;
    return result == expected;
  }
};

namespace {
const InType kInput3x3 = std::make_tuple(std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80, 90}, 3, 3);
const OutType kOutput3x3 = {20, 27, 35, 42, 50, 57, 65, 72, 80};

const InType kInput4x4 =
    std::make_tuple(std::vector<int>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150}, 4, 4);
const OutType kOutput4x4 = {12, 20, 30, 37, 42, 50, 60, 67, 82, 90, 100, 107, 112, 120, 130, 137};

const std::array<TestDS, 2> kTestData = {TestDS{kInput3x3, kOutput3x3}, TestDS{kInput4x4, kOutput4x4}};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<GaussFilterSEQ, InType>(kTestData, PPC_SETTINGS_makovskiy_i_gauss_filter_vert),
    ppc::util::AddFuncTask<GaussFilterMPI, InType>(kTestData, PPC_SETTINGS_makovskiy_i_gauss_filter_vert));

struct NameGenerator {
  // NOLINTNEXTLINE(readability-identifier-naming, readability-named-parameter)
  std::string operator()(const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestDS>> &info) const {
    auto name = std::get<1>(info.param);
    return name + "_" + std::to_string(info.index);
  }
};
}  // namespace

// NOLINTBEGIN(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(GaussFilterFuncTests, GaussFilterFuncTests, ppc::util::ExpandToValues(kTasks),
                         NameGenerator());
// NOLINTEND(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace)

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_P(GaussFilterFuncTests, RunFuncTests) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto task = std::get<0>(GetParam())(GetTestInputData());

  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  if (rank == 0) {
    auto result = task->GetOutput();
    const auto &expected = std::get<2>(GetParam()).second;
    ASSERT_EQ(result, expected);
  }
}
