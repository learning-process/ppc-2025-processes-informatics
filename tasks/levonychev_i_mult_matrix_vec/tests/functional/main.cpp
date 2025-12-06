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

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"
#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

class LevonychevIMultMatrixVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int mode = std::get<0>(params);
    std::vector<int64_t> matrix;
    switch (mode)
    {
    case 0:
      matrix = {
          1, 2, 3, 4, 5, 6,
          7, 8, 9, 10, 11, 12,
          13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36
      };
      input_data_ = std::make_tuple(matrix, 6, 6, std::vector<int64_t>{1, 1, 1, 1, 1, 1});
      output_data_ = std::vector<int64_t>(6);
      for (int i = 0; i < 6; ++i) {
        output_data_[i] = std::accumulate(matrix.begin() + i * 6, matrix.begin() + (i + 1) * 6, 0);
      }
      break;
    case 1:
      matrix = {
          1, 2,
          3, 4,
          5, 6,
          7, 8
      };
      input_data_ = std::make_tuple(matrix, 4, 2, std::vector<int64_t>{1, 1});
      output_data_ = std::vector<int64_t>{3, 7, 11, 15};
      break;
    case 2:
      matrix = {
          1, 2, 3, 4,
          5, 6, 7, 8
      };
      input_data_ = std::make_tuple(matrix, 2, 4, std::vector<int64_t>{1, 1, 1, 1});
      output_data_ = std::vector<int64_t>{10, 26};
      break;
    default:
      break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

namespace {

TEST_P(LevonychevIMultMatrixVecFuncTests, MultMatrixVec) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(0, "Square6_6"), std::make_tuple(1, "4_2"), std::make_tuple(2, "2_4")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LevonychevIMultMatrixVecMPI, InType>(kTestParam, PPC_SETTINGS_levonychev_i_mult_matrix_vec),
                   ppc::util::AddFuncTask<LevonychevIMultMatrixVecSEQ, InType>(kTestParam, PPC_SETTINGS_levonychev_i_mult_matrix_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LevonychevIMultMatrixVecFuncTests::PrintFuncTestName<LevonychevIMultMatrixVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(MultMatrixVecTests, LevonychevIMultMatrixVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace levonychev_i_mult_matrix_vec
