#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "luzan_e_simps_int/common/include/common.hpp"
#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"
#include "luzan_e_simps_int/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int n = get<0>(test_param);
    unsigned a = static_cast<unsigned>(100.0 * fabs(std::get<0>(std::get<1>(test_param))));
    unsigned b = static_cast<unsigned>(100.0 * fabs(std::get<1>(std::get<1>(test_param))));
    unsigned c = static_cast<unsigned>(100.0 * fabs(std::get<0>(std::get<2>(test_param))));
    unsigned d = static_cast<unsigned>(100.0 * fabs(std::get<1>(std::get<2>(test_param))));
    int func_num = std::get<3>(test_param);

    std::string square = "square_" + std::to_string(a) + "_" + std::to_string(b) + "___" + std::to_string(c) + "_" + std::to_string(d);
    return "for_" + std::to_string(n) + "pieces_" + "_" + square + "__by_func_No_" + std::to_string(func_num);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int n = std::get<0>(params);
    auto ab = std::get<1>(params);
    auto cd = std::get<2>(params);
    int fn = std::get<3>(params);

    input_data_ = std::make_tuple(n, ab, cd, fn);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    int n = 0;  // кол-во отрезков
    int func_num = 0;

    // getting data
    n = std::get<0>(input_data_);
    a = std::get<0>(std::get<1>(input_data_));
    b = std::get<1>(std::get<1>(input_data_));
    c = std::get<0>(std::get<2>(input_data_));
    d = std::get<1>(std::get<2>(input_data_));
    func_num = std::get<3>(input_data_);

    double (*fp)(double, double) = GetFunc(func_num);
    double hx = (b - a) / n;
    double hy = (d - c) / n;

    double sum = 0;
    double wx = 1.0;
    double wy = 1.0;
    double x = 0.0;
    double y = 0.0;

    for (int i = 0; i <= n; i++) {
      x = a + (hx * i);
      wx = GetWeight(i, n);

      for (int j = 0; j <= n; j++) {
        y = c + (hy * j);
        wy = GetWeight(j, n);
        sum += wy * wx * fp(x, y);
      }
    }
    sum = sum * hx * hy / (3 * 3);
    return (fabs(output_data - sum) < kEpsilon);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LuzanESimpsIntFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {std::make_tuple(4, std::make_tuple(3, 5), std::make_tuple(3, 5), 1),
                                            std::make_tuple(2, std::make_tuple(0, 1), std::make_tuple(0, 1), 1),
                                            std::make_tuple(10, std::make_tuple(0.5, 1), std::make_tuple(0.5, 1), 2),
                                            std::make_tuple(34, std::make_tuple(-50, 50), std::make_tuple(-50, 50), 1),
                                            std::make_tuple(42, std::make_tuple(-100, 0), std::make_tuple(0, 10), 1),
                                            std::make_tuple(22, std::make_tuple(-10, 0), std::make_tuple(0, 100), 1),
                                            std::make_tuple(6, std::make_tuple(3, 5), std::make_tuple(3, 5), 2),
                                            std::make_tuple(24, std::make_tuple(-2, 5), std::make_tuple(12, 25), 3),
                                            std::make_tuple(30, std::make_tuple(-20, 5), std::make_tuple(2, 25), 4)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LuzanESimpsIntMPI, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int),
                   ppc::util::AddFuncTask<LuzanESimpsIntSEQ, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuzanESimpsIntFuncTests::PrintFuncTestName<LuzanESimpsIntFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, LuzanESimpsIntFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luzan_e_simps_int
