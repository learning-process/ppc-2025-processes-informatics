#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "telnov_strongin_algorithm/common/include/common.hpp"
#include "telnov_strongin_algorithm/mpi/include/ops_mpi.hpp"
#include "telnov_strongin_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace telnov_strongin_algorithm {

class TelnovStronginAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    static const std::vector<InType> kTestCases = {{0.0, 1.0, 1e-1}};
    static size_t idx = 0;
    input_data_ = kTestCases[idx % kTestCases.size()];
    ++idx;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(TelnovStronginAlgorithmFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TelnovStronginAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_telnov_strongin_algorithm),
    ppc::util::AddFuncTask<TelnovStronginAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_telnov_strongin_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TelnovStronginAlgorithmFuncTests::PrintFuncTestName<TelnovStronginAlgorithmFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TelnovStronginAlgorithmFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace telnov_strongin_algorithm
