#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "smyshlaev_a_gauss_filt/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_gauss_filt/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace smyshlaev_a_gauss_filt {

class SmyshlaevAGaussFiltRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const ImageType &img = std::get<0>(test_param);
    return std::to_string(img.width) + "x" + std::to_string(img.height) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_.width == output_data.width && input_data_.height == output_data.height &&
            input_data_.channels == output_data.channels && !output_data.data.empty());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

ImageType CreateTestImage(int size) {
  ImageType img;
  img.width = size;
  img.height = size;
  img.channels = 3;
  img.data.resize(size * size * 3);

  for (size_t i = 0; i < img.data.size(); ++i) {
    img.data[i] = static_cast<uint8_t>(i % 256);
  }

  return img;
}

TEST_P(SmyshlaevAGaussFiltRunFuncTestsProcesses, GaussFiltTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(CreateTestImage(16), "16x16"),
                                            std::make_tuple(CreateTestImage(32), "32x32"),
                                            std::make_tuple(CreateTestImage(64), "64x64")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SmyshlaevAGaussFiltMPI, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_gauss_filt),
    ppc::util::AddFuncTask<SmyshlaevAGaussFiltSEQ, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_gauss_filt));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SmyshlaevAGaussFiltRunFuncTestsProcesses::PrintFuncTestName<SmyshlaevAGaussFiltRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussFiltTests, SmyshlaevAGaussFiltRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smyshlaev_a_gauss_filt
