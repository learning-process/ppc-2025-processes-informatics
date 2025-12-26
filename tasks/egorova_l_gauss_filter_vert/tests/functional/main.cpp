#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <tuple>
#include <array>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace egorova_l_gauss_filter_vert {

// ПЕРЕИМЕНОВАНО: EgorovaLRunFuncTestsProcesses -> EgorovaLGaussFilterRunFuncTests
class EgorovaLGaussFilterRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + 
           std::to_string(std::get<1>(test_param)) + "_" +
           std::to_string(std::get<2>(test_param)) + "_" + 
           std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    
    int r = std::get<0>(test_params);
    int c = std::get<1>(test_params);
    int ch = std::get<2>(test_params);

    input_data_.rows = r;
    input_data_.cols = c;
    input_data_.channels = ch;
    input_data_.data.assign(r * c * ch, 128); 
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.rows == input_data_.rows &&
           output_data.cols == input_data_.cols &&
           output_data.data.size() == input_data_.data.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

// ПЕРЕИМЕНОВАНО: Используем новый класс фикстуры
TEST_P(EgorovaLGaussFilterRunFuncTests, GaussFilterVertical) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(3, 3, 1, "small_3x3_gray"),
    std::make_tuple(10, 10, 3, "color_10x10_rgb"),
    std::make_tuple(20, 15, 1, "non_square_20x15"),
    std::make_tuple(5, 40, 1, "wide_image"),
    std::make_tuple(40, 5, 1, "tall_image")
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EgorovaLGaussFilterVertMPI, InType>(
        kTestParam, PPC_SETTINGS_egorova_l_gauss_filter_vert),
    ppc::util::AddFuncTask<EgorovaLGaussFilterVertSEQ, InType>(
        kTestParam, PPC_SETTINGS_egorova_l_gauss_filter_vert)
);
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = EgorovaLGaussFilterRunFuncTests::PrintFuncTestName<EgorovaLGaussFilterRunFuncTests>;

// ПЕРЕИМЕНОВАНО: GaussFilterTests и EgorovaLGaussFilterRunFuncTests
INSTANTIATE_TEST_SUITE_P(EgorovaLGaussFilterFunc, EgorovaLGaussFilterRunFuncTests, kGtestValues, kFuncTestName);

}  // namespace
}  // namespace egorova_l_gauss_filter_vert