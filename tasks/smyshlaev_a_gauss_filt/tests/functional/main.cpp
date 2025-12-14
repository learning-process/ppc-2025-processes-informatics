#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.width != output_data.width || input_data_.height != output_data.height ||
        input_data_.channels != output_data.channels || (input_data_.data.empty() && !output_data.data.empty()) ||
        (!input_data_.data.empty() && output_data.data.empty())) {
      return false;
    }
    if (input_data_.data.empty()) {
      return true;
    }

    OutType expected_output;
    auto seq_task = std::make_shared<SmyshlaevAGaussFiltSEQ>(input_data_);

    if (!seq_task->Validation()) {
      return false;
    }
    seq_task->PreProcessing();
    seq_task->Run();
    seq_task->PostProcessing();

    expected_output = seq_task->GetOutput();

    return expected_output.data == output_data.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(SmyshlaevAGaussFiltRunFuncTestsProcesses, GaussFiltTest) {
  ExecuteTest(GetParam());
}

ImageType CreateTestImage(int width, int height, int channels) {
  ImageType img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data.resize(static_cast<size_t>(width) * height * channels);

  for (size_t i = 0; i < img.data.size(); ++i) {
    img.data[i] = static_cast<uint8_t>(i % 256);
  }
  return img;
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(CreateTestImage(3, 3, 3), "3x3_RGB"),
    std::make_tuple(CreateTestImage(32, 32, 3), "32x32_RGB"),
    std::make_tuple(CreateTestImage(60, 20, 3), "60x20_RGB"),
    std::make_tuple(CreateTestImage(20, 60, 3), "20x60_RGB"),
    std::make_tuple(CreateTestImage(23, 31, 3), "23x31_RGB_PrimeDims"),
    std::make_tuple(CreateTestImage(40, 40, 1), "40x40_Grayscale"),
    std::make_tuple(CreateTestImage(24, 24, 4), "24x24_RGBA"),
    std::make_tuple(CreateTestImage(2, 2, 3), "2x2_ForceSequential"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SmyshlaevAGaussFiltMPI, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_gauss_filt),
    ppc::util::AddFuncTask<SmyshlaevAGaussFiltSEQ, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_gauss_filt));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName =
    SmyshlaevAGaussFiltRunFuncTestsProcesses::PrintFuncTestName<SmyshlaevAGaussFiltRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussFiltTests, SmyshlaevAGaussFiltRunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace smyshlaev_a_gauss_filt
