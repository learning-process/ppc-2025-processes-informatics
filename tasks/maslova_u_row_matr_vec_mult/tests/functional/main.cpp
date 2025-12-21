#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"
#include "maslova_u_row_matr_vec_mult/mpi/include/ops_mpi.hpp"
#include "maslova_u_row_matr_vec_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace maslova_u_row_matr_vec_mult {

class MaslovaURowMatrVecMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  MaslovaURowMatrVecMultFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_[i]) > 1e-6) {
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
  OutType expected_;
};

namespace {

TEST_P(MaslovaURowMatrVecMultFuncTests, matrVecMult) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    {std::make_tuple(InType{Matrix{{1, 2, 3, 4}, 2, 2}, {1, 1}}, OutType{3, 7}, "square_2x2"),
     std::make_tuple(InType{Matrix{{1, 0, 0, 1}, 2, 2}, {5, 3}}, OutType{5, 3}, "identity_2x2"),
     std::make_tuple(InType{Matrix{{1, 2, 3, 4, 5, 6}, 2, 3}, {1, 1, 1}}, OutType{6, 15}, "horizontal_2x3"),
     std::make_tuple(InType{Matrix{{1, 2, 3, 4, 5, 6}, 3, 2}, {1, 0}}, OutType{1, 3, 5}, "vertical_3x2"),
     std::make_tuple(InType{Matrix{{0, 0, 0, 0}, 2, 2}, {1, 2}}, OutType{0, 0}, "zero_matrix"),
     std::make_tuple(InType{Matrix{{-1, 2, -3, 4}, 2, 2}, {1, 2}}, OutType{3, 5}, "negative_values"),
     std::make_tuple(InType{Matrix{{2.5, 1.5, 4.0, 2.0}, 2, 2}, {2.0, 4.0}}, OutType{11.0, 16.0}, "double_values")}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MaslovaURowMatrVecMultMPI, InType>(kTestParam, PPC_SETTINGS_maslova_u_row_matr_vec_mult),
    ppc::util::AddFuncTask<MaslovaURowMatrVecMultSEQ, InType>(kTestParam, PPC_SETTINGS_maslova_u_row_matr_vec_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestName = MaslovaURowMatrVecMultFuncTests::PrintFuncTestName<MaslovaURowMatrVecMultFuncTests>;
INSTANTIATE_TEST_SUITE_P(matrVecMultTests, MaslovaURowMatrVecMultFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace maslova_u_row_matr_vec_mult
