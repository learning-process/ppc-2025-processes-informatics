#include <gtest/gtest.h>
#include <mpi.h>

#include <array>  // NOLINT(misc-include-cleaner)
#include <string>

#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"  // NOLINT(misc-include-cleaner)
#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"  // NOLINT(misc-include-cleaner)
#include "util/include/func_test_util.hpp"

namespace makovskiy_i_gauss_filter_vert {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestDS = std::pair<InType, OutType>;

class GaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestDS> {
 protected:
  InType GetTestInputData() override {  // NOLINT(readability-convert-member-functions-to-static)
    return std::get<2>(GetParam()).first;
  }

  bool CheckTestOutputData(OutType &result) override {  // NOLINT(readability-convert-member-functions-to-static)
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
}  // namespace

}  // namespace makovskiy_i_gauss_filter_vert

namespace {

using makovskiy_i_gauss_filter_vert::GaussFilterFuncTests;
using makovskiy_i_gauss_filter_vert::InType;
using makovskiy_i_gauss_filter_vert::kTestData;
using makovskiy_i_gauss_filter_vert::OutType;
using makovskiy_i_gauss_filter_vert::TestDS;

const auto kTasks = std::tuple_cat(ppc::util::AddFuncTask<makovskiy_i_gauss_filter_vert::GaussFilterSEQ, InType>(
                                       kTestData, PPC_SETTINGS_makovskiy_i_gauss_filter_vert),
                                   ppc::util::AddFuncTask<makovskiy_i_gauss_filter_vert::GaussFilterMPI, InType>(
                                       kTestData, PPC_SETTINGS_makovskiy_i_gauss_filter_vert));

struct NameGenerator {
  using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestDS>;
  using TestInfo = testing::TestParamInfo<FuncParam>;

  std::string operator()(const TestInfo &test_info) const {
    auto name = std::get<1>(test_info.param);
    return name + "_" + std::to_string(test_info.index);
  }
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(GaussFilterFuncTests, GaussFilterFuncTests, ppc::util::ExpandToValues(kTasks),
                         NameGenerator());
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)

TEST_P(GaussFilterFuncTests, RunFuncTests) {  // NOLINT(readability-function-cognitive-complexity)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto task = std::get<0>(GetParam())(GetTestInputData());

  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  if (rank == 0) {
    auto result = task->GetOutput();
    ASSERT_TRUE(CheckTestOutputData(result));
  }
}

}  // namespace
