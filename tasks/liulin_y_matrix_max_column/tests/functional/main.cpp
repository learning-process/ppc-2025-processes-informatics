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

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" +
           std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;

    {
      std::string abs_path =
          ppc::util::GetAbsoluteTaskPath(PPC_ID_liulin_y_matrix_max_column,
                                         "pic.jpg");
      auto *data =
          stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
      if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " +
                                 std::string(stbi_failure_reason()));
      }
      channels = STBI_rgb;
      img = std::vector<uint8_t>(
          data,
          data + static_cast<ptrdiff_t>(width * height * channels));
      stbi_image_free(data);

      if (width != height) {
        throw std::runtime_error("Test requires square image");
      }
    }

    const int N = width;

    // ---------- Создание матрицы (InType) ----------
    input_data_.assign(N, std::vector<int>(N, 0));

    // Комбинируем 3 RGB канала → одно число (например сумма)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int idx = (i * N + j) * channels;
        int pixel = img[idx] + img[idx + 1] + img[idx + 2];
        input_data_[i][j] = pixel;
      }
    }

    // ---------- Ожидаемый результат (OutType) ----------
    expected_output_.assign(N, 0);
    for (int col = 0; col < N; col++) {
      int mx = input_data_[0][col];
      for (int row = 1; row < N; row++) {
        mx = std::max(mx, input_data_[row][col]);
      }
      expected_output_[col] = mx;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(LiulinYMatrixMaxColumnFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, "3"),
    std::make_tuple(5, "5"),
    std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LiulinYMatrixMaxColumnMPI, InType>(
        kTestParam, PPC_SETTINGS_liulin_y_matrix_max_column),
    ppc::util::AddFuncTask<LiulinYMatrixMaxColumnSEQ, InType>(
        kTestParam, PPC_SETTINGS_liulin_y_matrix_max_column));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LiulinYMatrixMaxColumnFuncTests::PrintFuncTestName<
        LiulinYMatrixMaxColumnFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests,
                         LiulinYMatrixMaxColumnFuncTests,
                         kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace liulin_y_matrix_max_column
