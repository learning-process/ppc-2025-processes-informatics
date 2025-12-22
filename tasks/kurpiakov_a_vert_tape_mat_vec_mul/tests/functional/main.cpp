#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<2>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (size_t i = 0; i < expected_data_.size(); ++i) {
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
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(KurpiakovARunFuncTestsProcesses2, MatmulFromPic) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 10> kTestParam = {
    {std::make_tuple(std::make_tuple(0, std::vector<int>{}, std::vector<int>{}), "test1_empty",
                     std::vector<long long>{}),
     std::make_tuple(std::make_tuple(1, std::vector<int>{2}, std::vector<int>{3}), "test2_1x1",
                     std::vector<long long>{6}),
     std::make_tuple(std::make_tuple(2, std::vector<int>{1, 2, 3, 4}, std::vector<int>{5, 6}), "test3_2x2",
                     std::vector<long long>{17, 39}),
     std::make_tuple(std::make_tuple(3, std::vector<int>{1, 0, 0, 0, 1, 0, 0, 0, 1}, std::vector<int>{2, 3, 4}),
                     "test4_identity_3x3", std::vector<long long>{2, 3, 4}),
     std::make_tuple(std::make_tuple(2, std::vector<int>{0, 0, 0, 0}, std::vector<int>{1, 2}), "test5_zero_matrix",
                     std::vector<long long>{0, 0}),
     std::make_tuple(std::make_tuple(2, std::vector<int>{1, 2, 3, 4}, std::vector<int>{0, 0}), "test6_zero_vector",
                     std::vector<long long>{0, 0}),
     std::make_tuple(std::make_tuple(2, std::vector<int>{1, -2, -3, 4}, std::vector<int>{2, 1}),
                     "test7_negative_elements", std::vector<long long>{0, -2}),
     std::make_tuple(
         std::make_tuple(2, std::vector<int>{1000000, 2000000, 3000000, 4000000}, std::vector<int>{500, 600}),
         "test8_large_values", std::vector<long long>{1700000000LL, 3900000000LL}),
     std::make_tuple(std::make_tuple(3, std::vector<int>{2, 1, 1, 1, 2, 1, 1, 1, 2}, std::vector<int>{1, 1, 1}),
                     "test9_symmetric_matrix", std::vector<long long>{4, 4, 4}),
     std::make_tuple(std::make_tuple(2, std::vector<int>{10, 20, 30, 40}, std::vector<int>{2, 4}),
                     "test10_different_values", std::vector<long long>{100, 220})}};

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
