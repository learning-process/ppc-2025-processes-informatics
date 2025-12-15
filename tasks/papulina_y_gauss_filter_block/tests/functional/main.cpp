#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"
#include "papulina_y_gauss_filter_block/mpi/include/ops_mpi.hpp"
#include "papulina_y_gauss_filter_block/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace papulina_y_gauss_filter_block {

class PapulinaYFuncTestsGaussFilter : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name = std::get<0>(params);
    int width = std::get<1>(params);
    int height = std::get<2>(params);
    int channels = std::get<3>(params);

    std::mt19937 gen(std::hash<std::string>{}(test_name));
    std::uniform_int_distribution<> dist(0, 255);

    std::vector<unsigned char> pixels(static_cast<size_t>(width * height * channels));
    for (int i = 0; i < width * height * channels; i++) {
      pixels[i] = static_cast<unsigned char>(dist(gen));
    }

    input_data_ = Picture{width, height, channels, pixels};

    PapulinaYGaussFilterSEQ seq_filter(input_data_);
    seq_filter.Validation();
    seq_filter.PreProcessing();
    seq_filter.Run();
    seq_filter.PostProcessing();
    expected_result_ = seq_filter.GetOutput().pixels;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    /*std::cout << "Output: ";
    for(int i=0; i<output_data.pixels.size(); i++){
      std::cout << (int)output_data.pixels[i] << " ";
    }
    std::cout << std::endl;*/
    return (output_data.pixels == expected_result_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<unsigned char> expected_result_;
};

namespace {

TEST_P(PapulinaYFuncTestsGaussFilter, GaussFilterFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple("Random3x3BW", 3, 3, 1),     std::make_tuple("Random4x4BW", 4, 4, 1),
    std::make_tuple("Random5x5BW", 5, 5, 1),     std::make_tuple("Random6x6BW", 6, 6, 1),
    std::make_tuple("Random3x3RGB", 3, 3, 3),    std::make_tuple("Random4x4RGB", 4, 4, 3),
    std::make_tuple("Random5x5RGB", 5, 5, 3),    std::make_tuple("Random10x10BW", 10, 10, 1),
    std::make_tuple("Random15x15RGB", 15, 15, 3)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PapulinaYGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_papulina_y_gauss_filter_block),
    ppc::util::AddFuncTask<PapulinaYGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_papulina_y_gauss_filter_block));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PapulinaYFuncTestsGaussFilter::PrintFuncTestName<PapulinaYFuncTestsGaussFilter>;

INSTANTIATE_TEST_SUITE_P(GaussFilterFuncTests, PapulinaYFuncTestsGaussFilter, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace papulina_y_gauss_filter_block
