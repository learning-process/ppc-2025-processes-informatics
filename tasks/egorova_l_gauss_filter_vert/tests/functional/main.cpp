#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param)) + "_" +
           std::to_string(std::get<2>(test_param)) + "_" + std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int rr = std::get<0>(test_params);
    int cc = std::get<1>(test_params);
    int ch = std::get<2>(test_params);

    input_data_.rows = rr;
    input_data_.cols = cc;
    input_data_.channels = ch;
    input_data_.data.assign(static_cast<std::size_t>(rr) * static_cast<size_t>(cc) * static_cast<size_t>(ch), 128);
  }

  bool CheckTestOutputData(OutType& output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    return output_data.rows == input_data_.rows && output_data.cols == input_data_.cols &&
           output_data.data.size() == input_data_.data.size();
  }

  InType GetTestInputData() final { return input_data_; }

 private:
  InType input_data_;
};

namespace {
TEST_P(EgorovaLGaussFilterRunFuncTests, GaussFilterVertical) { ExecuteTest(GetParam()); }

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(3, 3, 1, "small_3x3_gray"), std::make_tuple(10, 10, 3, "color_10x10_rgb"),
    std::make_tuple(20, 15, 1, "non_square_20x15"), std::make_tuple(5, 40, 1, "wide_image"),
    std::make_tuple(40, 5, 1, "tall_image")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<EgorovaLGaussFilterVertMPI, InType>(kTestParam,
                                                                             PPC_SETTINGS_egorova_l_gauss_filter_vert),
                   ppc::util::AddFuncTask<EgorovaLGaussFilterVertSEQ, InType>(kTestParam,
                                                                             PPC_SETTINGS_egorova_l_gauss_filter_vert));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName = EgorovaLGaussFilterRunFuncTests::PrintFuncTestName<EgorovaLGaussFilterRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(EgorovaLGaussFilterFunc, EgorovaLGaussFilterRunFuncTests, kGtestValues, kFuncTestName);
}  // namespace
}  // namespace egorova_l_gauss_filter_vert