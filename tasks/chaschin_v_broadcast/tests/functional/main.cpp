#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "example_processes/common/include/common.hpp"
#include "example_processes/mpi/include/ops_mpi.hpp"
#include "example_processes/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_broadcast {

class ChaschinVRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_.resize(size);
    for (int i = 0; i < array_size; ++i) {
      input_data_[i] = (i * 187345543) % 100;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ChaschinVRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ChaschinVBroadcastMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
                   ppc::util::AddFuncTask<ChaschinVBroadcastSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChaschinVRunFuncTestsProcesses::PrintFuncTestName<ChaschinVRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ChaschinVRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chaschin_v_broadcast
