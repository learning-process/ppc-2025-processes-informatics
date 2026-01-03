#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace egorova_l_gauss_filter_vert {

using TestType = std::tuple<int, int, int, std::string>;

class EgorovaLGaussFilterRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  using ParamType = typename ppc::util::BaseRunFuncTests<InType, OutType, TestType>::ParamType;

  static std::string PrintTestParam(const ::testing::TestParamInfo<ParamType> &info) {
    const auto &params = info.param;
    const auto &test_params = std::get<2>(params);

    std::string name = std::get<1>(params) + "_" + std::to_string(std::get<0>(test_params)) + "_" +
                       std::to_string(std::get<1>(test_params)) + "_" + std::to_string(std::get<2>(test_params)) + "_" +
                       std::get<3>(test_params);

    std::ranges::replace(name, '-', '_');
    std::ranges::replace(name, ' ', '_');
    return name;
  }

 protected:
  void SetUp() override {
    const auto &test_params = std::get<2>(GetParam());

    input_.rows = std::get<0>(test_params);
    input_.cols = std::get<1>(test_params);
    input_.channels = std::get<2>(test_params);

    size_t sz = static_cast<size_t>(input_.rows) * input_.cols * input_.channels;

    input_.data.resize(sz);
    for (size_t i = 0; i < sz; ++i) {
      input_.data[i] = static_cast<uint8_t>((i * 13) % 255);
    }
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) final {
    const auto &in = input_;
    if (out.rows != in.rows || out.cols != in.cols || out.channels != in.channels) {
      return false;
    }

    return out.data.size() == static_cast<size_t>(in.rows) * in.cols * in.channels;
  }

 private:
  InType input_;
};

TEST_P(EgorovaLGaussFilterRunFuncTests, GaussFilterVertical) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {
    std::make_tuple(3, 3, 1, "small_3x3_gray"), std::make_tuple(10, 10, 3, "color_10x10_rgb"),
    std::make_tuple(20, 15, 1, "non_square"),   std::make_tuple(5, 40, 1, "wide_image"),
    std::make_tuple(40, 5, 1, "tall_image"),
};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<EgorovaLGaussFilterVertSEQ, InType>(kTestParams, PPC_SETTINGS_egorova_l_gauss_filter_vert),
    ppc::util::AddFuncTask<EgorovaLGaussFilterVertMPI, InType>(kTestParams, PPC_SETTINGS_egorova_l_gauss_filter_vert));

INSTANTIATE_TEST_SUITE_P(EgorovaLGaussFilterFunc, EgorovaLGaussFilterRunFuncTests, ppc::util::ExpandToValues(kTasks),
                         EgorovaLGaussFilterRunFuncTests::PrintTestParam);

}  // namespace egorova_l_gauss_filter_vert
