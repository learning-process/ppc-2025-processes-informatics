#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int sz = 5;         // размер матрицы
    const int sizeOfRib = 2;  // полу-ширина + 1 (k = 1)
    input_data_ = GenerateTestInput(sz, sizeOfRib);

    return;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-6;

    const auto &matrix = input_data_.first;
    const auto &b = input_data_.second;
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[i][j] * output_data[j];
      }
      if (std::abs(sum - b[i]) > eps) {
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
  OutType expect_res_;
};

static std::pair<std::vector<std::vector<double>>, std::vector<double>> GenerateTestInput(int sz, int sizeOfRib) {
  std::vector<std::vector<double>> matrix(sz, std::vector<double>(sz, 0.0));
  std::vector<double> vec(sz, 0.0);

  int k = sizeOfRib - 1;

  for (int i = 0; i < sz; ++i) {
    double sum = 0.0;

    for (int j = std::max(0, i - k); j <= std::min(sz - 1, i + k); ++j) {
      if (i == j) {
        continue;
      }
      matrix[i][j] = -1.0;
      sum += std::abs(matrix[i][j]);
    }

    matrix[i][i] = sum + 1.0;  // диагональное преобладание
    vec[i] = 1.0;
  }

  return {matrix, vec};
}

namespace {

TEST_P(ShvetsovaKGaussVertStripRunFuncTestsProcesses, DataFromTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 0> kTestParam = {};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShvetsovaKGaussVertStripMPI, InType>(kTestParam, PPC_SETTINGS_shvetsova_k_gausse_vert_strip),
    ppc::util::AddFuncTask<ShvetsovaKGaussVertStripSEQ, InType>(kTestParam,
                                                                PPC_SETTINGS_shvetsova_k_gausse_vert_strip));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShvetsovaKGaussVertStripRunFuncTestsProcesses::PrintFuncTestName<ShvetsovaKGaussVertStripRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxDiffTest, ShvetsovaKGaussVertStripRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shvetsova_k_gausse_vert_strip
