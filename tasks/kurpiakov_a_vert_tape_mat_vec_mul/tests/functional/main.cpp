#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_vert_tape_mat_vec_mul/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

class KurpiakovARunFuncTestsProcesses2 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<2>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (int i = 0; i < static_cast<int>(expected_data_.size()); ++i) {
      if (expected_data_[i] != output_data[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_data_{};
};

namespace {

TEST_P(KurpiakovARunFuncTestsProcesses2, MatmulFromPic) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 10> kTestParam = {
    {std::make_tuple(std::make_tuple(0LL, std::vector<int64_t>{}, std::vector<int64_t>{}), "test1_empty",
                     std::vector<int64_t>{}),
     std::make_tuple(std::make_tuple(1LL, std::vector<int64_t>{2LL}, std::vector<int64_t>{3LL}), "test2_1x1",
                     std::vector<int64_t>{6LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{1LL, 2LL, 3LL, 4LL}, std::vector<int64_t>{5LL, 6LL}),
                     "test3_2x2", std::vector<int64_t>{17LL, 39LL}),
     std::make_tuple(std::make_tuple(3LL, std::vector<int64_t>{1LL, 0LL, 0LL, 0LL, 1LL, 0LL, 0LL, 0LL, 1LL},
                                     std::vector<int64_t>{2LL, 3LL, 4LL}),
                     "test4_identity_3x3", std::vector<int64_t>{2LL, 3LL, 4LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{0LL, 0LL, 0LL, 0LL}, std::vector<int64_t>{1LL, 2LL}),
                     "test5_zero_matrix", std::vector<int64_t>{0LL, 0LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{1LL, 2LL, 3LL, 4LL}, std::vector<int64_t>{0LL, 0LL}),
                     "test6_zero_vector", std::vector<int64_t>{0LL, 0LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{1LL, -2LL, -3LL, 4LL}, std::vector<int64_t>{2LL, 1LL}),
                     "test7_negative_elements", std::vector<int64_t>{0LL, -2LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{1000000LL, 2000000LL, 3000000LL, 4000000LL},
                                     std::vector<int64_t>{500LL, 600LL}),
                     "test8_large_values", std::vector<int64_t>{1700000000LL, 3900000000LL}),
     std::make_tuple(std::make_tuple(3LL, std::vector<int64_t>{2LL, 1LL, 1LL, 1LL, 2LL, 1LL, 1LL, 1LL, 2LL},
                                     std::vector<int64_t>{1LL, 1LL, 1LL}),
                     "test9_symmetric_matrix", std::vector<int64_t>{4LL, 4LL, 4LL}),
     std::make_tuple(std::make_tuple(2LL, std::vector<int64_t>{10LL, 20LL, 30LL, 40LL}, std::vector<int64_t>{2LL, 4LL}),
                     "test10_different_values", std::vector<int64_t>{100LL, 220LL})}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KurpiakovAVretTapeMulMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kurpiakov_a_vert_tape_mat_vec_mul),
                                           ppc::util::AddFuncTask<KurpiakovAVretTapeMulSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kurpiakov_a_vert_tape_mat_vec_mul));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KurpiakovARunFuncTestsProcesses2::PrintFuncTestName<KurpiakovARunFuncTestsProcesses2>;
// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(MatVecMulTests, KurpiakovARunFuncTestsProcesses2, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
