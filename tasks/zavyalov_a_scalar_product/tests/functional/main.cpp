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

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zavyalov_a_scalar_product/common/include/common.hpp"
#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_scalar_product/seq/include/ops_seq.hpp"

namespace zavyalov_a_scalar_product {

class ZavyalovAScalarProductFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::vector<double> leftVec(params);
    std::vector<double> rightVec(params);

    double minus = -1.0;
    for (unsigned int i = 0; i < params; i++) {
      leftVec[i] = i * 0.5 + 0.1;
      rightVec[i] = double(i) + 1.0;
      rightVec[i] *= minus;
      minus *= -1.0;
    }

    input_data_ = std::make_tuple(leftVec, rightVec);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double res = 0.0;
    for (size_t i = 0; i < std::get<0>(input_data_).size(); i++) {
      res += std::get<0>(input_data_)[i] * std::get<1>(input_data_)[i];
    }
    return (fabs(res - output_data) < 1e-6);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ZavyalovAScalarProductFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZavyalovAScalarProductMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_scalar_product),
    ppc::util::AddFuncTask<ZavyalovAScalarProductSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_scalar_product));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAScalarProductFuncTests::PrintFuncTestName<ZavyalovAScalarProductFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAScalarProductFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_scalar_product
