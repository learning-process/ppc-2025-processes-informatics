#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <cstdint>
#include <vector>



#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"
#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVRunFuncTestsProcessesSO : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int size = std::get<0>(params);

    // Доступ к tuple
    auto &image = std::get<0>(input_data_);  // std::vector<std::vector<Pixel>>
    auto &width = std::get<1>(input_data_);
    auto &height = std::get<2>(input_data_);

    // Задаём размеры
    width = size;
    height = size;

    // Генерация входного изображения (детерминированные пиксели)
    image.resize(height);
    for (int i = 0; i < height; ++i) {
      image[i].resize(width);
      for (int j = 0; j < width; ++j) {
        image[i][j] = Pixel{
    .r = static_cast<uint8_t>((i + 1) % 256),
    .g = static_cast<uint8_t>((j + 2) % 256),
    .b = static_cast<uint8_t>(((i + 4272) * (j + 422)) % 256)
};

      }
    }
    /*std::cout<<"Input: ";
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        std::cout<< "{" <<
    static_cast<int>(image[i][j].r)<<","<<static_cast<int>(image[i][j].g)<<","<<static_cast<int>(image[i][j].b)<< "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/

    std::vector<std::vector<float>> gray(height + 2, std::vector<float>(width + 2, 0.0F));
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        const auto &p = image[i][j];
        gray[i + 1][j + 1] = (0.299F * p.r) + (0.587F * p.g) + (0.114F * p.b);
      }
    }

    /*std::cout<<"gray: ";
    for (int i = 0; i < width+2; i++) {
      for (int j = 0; j < height+2; j++) {
        std::cout<<gray[i][j]<< ' ';
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/

    // Маски Sobel
    static constexpr std::array<std::array<int, 3>, 3> kKx{{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};

  static constexpr std::array<std::array<int, 3>, 3> kKy{{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

    // Подготовка expected_output_
    expected_output_.resize(height);
    for (int i = 0; i < height; ++i) {
      expected_output_[i].resize(width);
    }

    // Вычисление градиента Sobel и реконструкция Pixel
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        float gx = 0.0F;
        float gy = 0.0F;

        // С учётом паддинга в gray[width+2][height+2]
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            float val = gray[i + 1 + di][j + 1 + dj];  // +1 для сдвига на паддинг
            gx += val * static_cast<int>(kKx[di + 1][dj + 1]);
            gy += val * static_cast<int>(kKy[di + 1][dj + 1]);
          }
        }

        float grad = std::sqrt((gx * gx )+ (gy * gy));
        // std::cout << grad << "\n";

        uint8_t intensity = static_cast<uint8_t>(std::min(255.0F, grad));
        expected_output_[i][j] = Pixel{.r=intensity, .g=intensity, .b=intensity};
      }
    }

    /*std::cout<<"Output: ";
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        std::cout<< "{" <<
    static_cast<int>(expected_output_[i][j].r)<<","<<static_cast<int>(expected_output_[i][j].g)<<","<<static_cast<int>(expected_output_[i][j].b)<<
    "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/
  }

  bool CheckTestOutputData(OutType &output_data) final {
    /*std::cout << "Output check: ";
    for (size_t i = 0; i < output_data.size(); i++) {
      for (size_t j = 0; j < output_data[0].size(); j++) {
        std::cout << "{" << static_cast<int>(output_data[i][j].r) << "," << static_cast<int>(output_data[i][j].g) << ","
                  << static_cast<int>(output_data[i][j].b) << "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Exp check: ";
    for (size_t i = 0; i < expected_output_.size(); i++) {
      for (size_t j = 0; j < expected_output_[0].size(); j++) {
        std::cout << "{" << static_cast<int>(expected_output_[i][j].r) << ","
                  << static_cast<int>(expected_output_[i][j].g) << "," << static_cast<int>(expected_output_[i][j].b)
                  << "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/

    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ChaschinVRunFuncTestsProcessesSO, SobelOperator) {
  ExecuteTest(GetParam());
}

/*const std::array<TestType, 20> kTestParam = {
    std::make_tuple(4, "4"),   std::make_tuple(5, "5"),   std::make_tuple(7, "7"),   std::make_tuple(8, "8"),
    std::make_tuple(9, "9"),   std::make_tuple(10, "10"), std::make_tuple(11, "11"), std::make_tuple(13, "13"),
    std::make_tuple(15, "15"), std::make_tuple(17, "17"), std::make_tuple(19, "19"), std::make_tuple(21, "21"),
    std::make_tuple(23, "23"), std::make_tuple(25, "25"), std::make_tuple(27, "27"), std::make_tuple(29, "29"),
    std::make_tuple(31, "31"), std::make_tuple(33, "33"), std::make_tuple(35, "35"), std::make_tuple(37, "37")};*/

const std::array<TestType, 4> kTestParam = {std::make_tuple(4, "4"), std::make_tuple(13, "13"), std::make_tuple(7, "7"),
                                            std::make_tuple(8, "8")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChaschinVSobelOperatorMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<ChaschinVSobelOperatorSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChaschinVRunFuncTestsProcessesSO::PrintFuncTestName<ChaschinVRunFuncTestsProcessesSO>;

INSTANTIATE_TEST_SUITE_P(MatrixRowMaxTests, ChaschinVRunFuncTestsProcessesSO, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chaschin_v_sobel_operator
