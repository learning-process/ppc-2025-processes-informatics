#include <array>
#include <cctype>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

using ::testing::TestWithParam;

class YurkinCountingNumberFuncTests : public TestWithParam<std::tuple<InType, std::string, int>> {};

TEST_P(YurkinCountingNumberFuncTests, MainTest) {
  auto params = GetParam();
  const auto &input = std::get<0>(params);
  const auto &str = std::get<1>(params);
  const int expected = std::get<2>(params);

  InType modified = input;
  modified.insert(modified.end(), str.begin(), str.end());

  // Sequential
  YurkinCountingNumberSEQ seq_task(modified);
  seq_task.Validation();
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();
  OutType seq_res = seq_task.GetOutput();

  // MPI
  YurkinCountingNumberMPI mpi_task(modified);
  mpi_task.Validation();
  mpi_task.PreProcessing();
  mpi_task.Run();
  mpi_task.PostProcessing();
  OutType mpi_res = mpi_task.GetOutput();

  EXPECT_EQ(seq_res, expected);
  EXPECT_EQ(mpi_res, expected);
}

static const std::array<std::tuple<InType, std::string, int>, 3> kValues = {
    std::make_tuple(InType{}, std::string("AbC123"), 3), std::make_tuple(InType{}, std::string("aaaa"), 4),
    std::make_tuple(InType{'1', '2', '3'}, std::string("ABC"), 3)};

INSTANTIATE_TEST_SUITE_P(YurkinCountingTests, YurkinCountingNumberFuncTests, ::testing::ValuesIn(kValues),
                         ::testing::PrintToStringParamName());

}  // namespace yurkin_counting_number
