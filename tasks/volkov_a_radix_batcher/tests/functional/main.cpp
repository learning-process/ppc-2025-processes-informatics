#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
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

  void GenerateRandomData(size_t size, double min_val, double max_val) {
    test_input.resize(size);
    for (size_t i = 0; i < size; ++i) {
      double phase = std::sin((static_cast<double>(i) * 12.9898) + 42.0);
      double normalized = (phase + 1.0) / 2.0;
      test_input[i] = min_val + (normalized * (max_val - min_val));
    }
  }

  void SetUp() override {
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (param == "generate_empty") {
      test_input = {};
    } else if (param == "generate_single") {
      test_input = {3.0};
    } else if (param == "generate_all_equal") {
      test_input.assign(1000, 123.456);
    } else if (param == "generate_sorted") {
      test_input.resize(1000);
      double start_val = -500.0;
      for (size_t i = 0; i < test_input.size(); ++i) {
        test_input[i] = start_val + static_cast<double>(i);
      }
    } else if (param == "generate_reverse") {
      test_input.resize(1000);
      double start_val = -500.0;
      for (size_t i = 0; i < test_input.size(); ++i) {
        test_input[i] = start_val + static_cast<double>(i);
      }
      std::ranges::reverse(test_input);
    } else if (param == "generate_dense_alternating") {
      size_t size = 2000;
      test_input.resize(size);
      for (size_t i = 0; i < size; ++i) {
        auto val = static_cast<double>(i + 1);
        // Чередуем: +, -, +, - ...
        test_input[i] = (i % 2 == 0) ? val : -val;
      }
    } else if (param == "generate_zeros") {
      test_input.assign(1000, 0.0);
    } else if (param == "generate_negative_mixed") {
      test_input = {0.0, -0.0, 5.0, -5.0, 1.0, -1.0, 2.5, -2.5};
    } else if (param == "generate_extreme_values") {
      test_input = {
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::min(),
          std::numeric_limits<double>::epsilon(),
          0.0,
          -0.0,
          1.0,
          -1.0,
          1e100,
          -1e100,
      };
    } else if (param == "generate_random_small") {
      GenerateRandomData(17, -100.0, 100.0);
    } else if (param == "generate_random_medium") {
      GenerateRandomData(128, -1e5, 1e5);
    } else if (param == "generate_random_large") {
      GenerateRandomData(1000, -1e9, 1e9);
    } else {
      std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_volkov_a_radix_batcher, param + ".txt");
      std::ifstream stream(path);
      if (stream.is_open()) {
        size_t size = 0;
        stream >> size;
        test_input.resize(size);
        for (size_t i = 0; i < size; ++i) {
          stream >> test_input[i];
        }
      }
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

const std::array<TestType, 14> kTestParams = {"test1",
                                              "test2",
                                              "generate_empty",
                                              "generate_single",
                                              "generate_all_equal",
                                              "generate_sorted",
                                              "generate_reverse",
                                              "generate_zeros",
                                              "generate_dense_alternating",
                                              "generate_negative_mixed",
                                              "generate_extreme_values",
                                              "generate_random_small",
                                              "generate_random_medium",
                                              "generate_random_large"};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<VolkovARadixBatcherMPI, InType>(kTestParams, PPC_SETTINGS_volkov_a_radix_batcher),
    ppc::util::AddFuncTask<VolkovARadixBatcherSEQ, InType>(kTestParams, PPC_SETTINGS_volkov_a_radix_batcher));

const auto kGTestParams = ppc::util::ExpandToValues(kTasks);

INSTANTIATE_TEST_SUITE_P(RadixSortTests, VolkovARadixBatcherFuncTests, kGTestParams,
                         VolkovARadixBatcherFuncTests::PrintFuncTestName<VolkovARadixBatcherFuncTests>);

}  // namespace

}  // namespace volkov_a_radix_batcher
