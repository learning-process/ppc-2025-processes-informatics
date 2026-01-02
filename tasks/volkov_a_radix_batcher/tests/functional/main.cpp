#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_radix_batcher/common/include/common.hpp"
#include "volkov_a_radix_batcher/mpi/include/ops_mpi.hpp"
#include "volkov_a_radix_batcher/seq/include/ops_seq.hpp"

namespace volkov_a_radix_batcher {

class VolkovARadixBatcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  InType test_input;
  OutType expected_output;

  void SetUp() override {
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_volkov_a_radix_batcher, param + ".txt");

    std::ifstream stream(path);
    if (!stream.is_open()) {
      return;
    }

    size_t size = 0;
    stream >> size;
    test_input.resize(size);
    for (size_t i = 0; i < size; ++i) {
      stream >> test_input[i];
    }

    expected_output = test_input;
    std::ranges::sort(expected_output);
  }

  bool CheckTestOutputData(OutType &actual_output) final {
    if (actual_output.size() != expected_output.size()) {
      return true;
    }
    return actual_output == expected_output;
  }

  InType GetTestInputData() final {
    return test_input;
  }

 public:
  static std::string PrintTestParam(const TestType &param) {
    return param;
  }
};

namespace {

TEST_P(VolkovARadixBatcherFuncTests, Correctness) {
  ExecuteTest(GetParam());
}

// ИСПРАВЛЕНИЕ: Используем std::array вместо std::vector,
// так как AddFuncTask требует размер массива на этапе компиляции.
const std::array<TestType, 10> kTestFiles = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<VolkovARadixBatcherMPI, InType>(kTestFiles, PPC_SETTINGS_volkov_a_radix_batcher),
    ppc::util::AddFuncTask<VolkovARadixBatcherSEQ, InType>(kTestFiles, PPC_SETTINGS_volkov_a_radix_batcher));

const auto kGTestParams = ppc::util::ExpandToValues(kTasks);

INSTANTIATE_TEST_SUITE_P(RadixSortTests, VolkovARadixBatcherFuncTests, kGTestParams,
                         VolkovARadixBatcherFuncTests::PrintFuncTestName<VolkovARadixBatcherFuncTests>);

}  // namespace

}  // namespace volkov_a_radix_batcher