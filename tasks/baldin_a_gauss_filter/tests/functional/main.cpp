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

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "baldin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_gauss_filter {

namespace {

ImageData CalculateGaussFilter(const ImageData& src) {
  ImageData dst = src;
  int w = src.width;
  int h = src.height;
  int c = src.channels;
  
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  
  for(int y=0; y<h; ++y) {
      for(int x=0; x<w; ++x) {
          for(int k=0; k<c; ++k) {
              int sum = 0;
              for(int dy=-1; dy<=1; ++dy) {
                  for(int dx=-1; dx<=1; ++dx) {
                      int ny = std::clamp(y + dy, 0, h - 1);
                      int nx = std::clamp(x + dx, 0, w - 1);
                      
                      sum += src.pixels[(ny * w + nx) * c + k] * kernel[dy+1][dx+1];
                  }
              }
              dst.pixels[(y * w + x) * c + k] = static_cast<uint8_t>(sum / 16);
          }
      }
  }
  return dst;
}

}

class BaldinAGaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType filename = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam())  + ".jpg";
    //std::cout << "FILE: " << filename << '\n';
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_baldin_a_gauss_filter, filename);
    //std::cout << "ABS PATH: " << abs_path << '\n';
    auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 0);
    if (data == nullptr) {
      throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }
    img = std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width * height * channels)));
    stbi_image_free(data);

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = ImageData(width, height, channels, img);
    expected_output_ = CalculateGaussFilter(input_data_);
    //for (int i = 0; i < 12; i++) std::cout << (int)(expected_output_.pixels[i]) << ' ';
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // for (int i = 0; i < 1; i++) {
    //   std::cout << expected_output_.pixels[i] << " : " << output_data.pixels[i] << '\n';
    // }
    return (output_data == expected_output_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(BaldinAGaussFilterFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = { "pic", "image", "image1" };

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_gauss_filter),
                   ppc::util::AddFuncTask<BaldinAGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAGaussFilterFuncTests::PrintFuncTestName<BaldinAGaussFilterFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussFilterTests, BaldinAGaussFilterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_gauss_filter
