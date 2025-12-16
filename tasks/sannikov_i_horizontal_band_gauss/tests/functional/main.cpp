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

#include "sannikov_i_horizontal_band_gauss/common/include/common.hpp"
#include "sannikov_i_horizontal_band_gauss/mpi/include/ops_mpi.hpp"
#include "sannikov_i_horizontal_band_gauss/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_horizontal_band_gauss {

class SannikovIHorizontalBandGaussFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &A = std::get<0>(input_data_);
    const auto &b = std::get<1>(input_data_);

    if (A.empty() || b.empty()) {
      return false;
    }

    const std::size_t n = A.size();
    if (output_data.size() != n) {
      return false;
    }

    double max_abs_residual = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        s += A[i][j] * output_data[j];
      }
      const double r = abs(s - b[i]);
      if (r > max_abs_residual) {
        max_abs_residual = r;
      }
    }

    return max_abs_residual < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

static std::vector<std::vector<double>> MakeTridiagonal(std::size_t n, double a, double b, double c) {
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    A[i][i] = b;
    if (i > 0) {
      A[i][i - 1] = a;
    }
    if (i + 1 < n) {
      A[i][i + 1] = c;
    }
  }
  return A;
}

TEST_P(SannikovIHorizontalBandGaussFuncTests, SolveFromParams) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(InType{std::vector<std::vector<double>>{{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 5.0}},
                           std::vector<double>{2.0, 8.0, 10.0}, std::size_t{0}},
                    "diag_3_band0"),

    std::make_tuple(InType{std::vector<std::vector<double>>{
                               {3.0, 0.0, 0.0, 0.0}, {0.0, 3.0, 0.0, 0.0}, {0.0, 0.0, 3.0, 0.0}, {0.0, 0.0, 0.0, 3.0}},
                           std::vector<double>{3.0, 6.0, 9.0, 12.0}, std::size_t{10}},
                    "diag_4_band_big"),

    std::make_tuple(
        InType{MakeTridiagonal(5, -1.0, 4.0, -1.0), std::vector<double>{3.0, 2.0, 2.0, 2.0, 3.0}, std::size_t{1}},
        "tridiag_5_band1"),

    std::make_tuple(InType{MakeTridiagonal(6, 1.0, 5.0, 1.0), std::vector<double>{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0},
                           std::size_t{1}},
                    "tridiag_6_band1_negative_b"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{7.0}}, std::vector<double>{14.0}, std::size_t{0}}, "1x1"),

    std::make_tuple(
        InType{std::vector<std::vector<double>>{{2.0, 1.0}, {1.0, 3.0}}, std::vector<double>{3.0, 4.0}, std::size_t{1}},
        "2x2_band1"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{10.0, 2.0, 0.0}, {3.0, 9.0, 4.0}, {0.0, 1.0, 7.0}},
                           std::vector<double>{12.0, 16.0, 8.0}, std::size_t{1}},
                    "3x3_band1"),

    std::make_tuple(InType{std::vector<std::vector<double>>{
                               {5.0, 1.0, 2.0, 0.0}, {1.0, 6.0, 1.0, 3.0}, {0.0, 2.0, 7.0, 1.0}, {0.0, 0.0, 4.0, 8.0}},
                           std::vector<double>{8.0, 11.0, 9.0, 12.0}, std::size_t{2}},
                    "4x4_band2"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{1.0, 0.0, 0.0, 0.0, 0.0},
                                                            {0.0, 2.0, 0.0, 0.0, 0.0},
                                                            {0.0, 0.0, 3.0, 0.0, 0.0},
                                                            {0.0, 0.0, 0.0, 4.0, 0.0},
                                                            {0.0, 0.0, 0.0, 0.0, 5.0}},
                           std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}, std::size_t{0}},
                    "diag_5_band0"),

    std::make_tuple(
        InType{MakeTridiagonal(5, -1.0, 10.0, -1.0), std::vector<double>{9.0, 8.0, 7.0, 6.0, 5.0}, std::size_t{1}},
        "tridiag_5_strong_diag"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{2.5, 1.0, 0.0}, {1.0, 3.5, 1.0}, {0.0, 1.0, 4.5}},
                           std::vector<double>{3.5, 5.5, 6.5}, std::size_t{1}},
                    "fractions_3x3"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{9.0, 1.0, 1.0, 0.0, 0.0, 0.0},
                                                            {2.0, 10.0, 1.0, 1.0, 0.0, 0.0},
                                                            {0.0, 3.0, 11.0, 1.0, 1.0, 0.0},
                                                            {0.0, 0.0, 4.0, 12.0, 1.0, 1.0},
                                                            {0.0, 0.0, 0.0, 5.0, 13.0, 1.0},
                                                            {0.0, 0.0, 0.0, 0.0, 6.0, 14.0}},
                           std::vector<double>{11.0, 14.0, 16.0, 18.0, 19.0, 20.0}, std::size_t{2}},
                    "6x6_band2")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SannikovIHorizontalBandGaussMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sannikov_i_horizontal_band_gauss),
                                           ppc::util::AddFuncTask<SannikovIHorizontalBandGaussSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sannikov_i_horizontal_band_gauss));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SannikovIHorizontalBandGaussFuncTests::PrintFuncTestName<SannikovIHorizontalBandGaussFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SannikovIHorizontalBandGaussFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sannikov_i_horizontal_band_gauss
