#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luzan_e_simps_int/common/include/common.hpp"
#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"
#include "luzan_e_simps_int/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    double n = std::get<0>(params);
    auto ab = std::get<1>(params);
    auto cd = std::get<2>(params);

    input_data_ = std::make_tuple(n, ab, cd);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double a = 0.0, b = 0.0;
    double c = 0.0, d = 0.0;
    int n = 0;  // кол-во отрезков

    // getting data
    n = std::get<0>(input_data_);
    a = std::get<0>(std::get<1>(input_data_));
    b = std::get<1>(std::get<1>(input_data_));
    c = std::get<0>(std::get<2>(input_data_));
    d = std::get<1>(std::get<2>(input_data_));

    double hx = (b - a) / n;
    double hy = (d - c) / n;

    double sum = 0;
    double wx = 1.0, wy = 1.0;
    double x = 0.0, y = 0.0;

    for (int i = 0; i <= n; i++) {
      x = a + hx * i;
      wx = 1.0;
      if (i != 0 && i != n) {
        if (i % 2 == 1) {
          wx = 4.0;
        } else {
          wx = 2.0;
        }
      }

      for (int j = 0; j <= n; j++) {
        wy = 1.0;
        y = c + hy * j;

        if (j != 0 && j != n) {
          if (j % 2 == 1) {
            wy = 4.0;
          } else {
            wy = 2.0;
          }
        }
        sum += wy * wx * f(x, y);
      }
    }
    sum = sum * hx * hy / (3 * 3);

    return (abs(output_data - sum) < EPSILON);
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

const std::array<TestType, 1> kTestParam = {std::make_tuple(10, std::make_tuple(3, 5), std::make_tuple(3, 5))};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LuzanESimpsIntMPI, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int),
                   ppc::util::AddFuncTask<LuzanESimpsIntSEQ, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuzanESimpsIntFuncTests::PrintFuncTestName<LuzanESimpsIntFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, LuzanESimpsIntFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luzan_e_simps_int
