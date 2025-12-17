#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"
#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"
#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_image_smooth {

class GaseninLRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int kernel_size = std::get<0>(test_params);
    std::string test_name = std::get<1>(test_params);

    if (test_name == "tiny_image_for_coverage") {
      input_data_.width = 1;
      input_data_.height = 1;
      input_data_.kernel_size = 3;
      input_data_.data = {255};
    }
    if (test_name == "small_image") {
      input_data_.width = 10;
      input_data_.height = 2;
      input_data_.data.resize(static_cast<size_t>(input_data_.width) * static_cast<size_t>(input_data_.height));

      for (size_t i = 0; i < input_data_.data.size(); ++i) {
        input_data_.data[i] = static_cast<uint8_t>(i % 256);
      }
    } else {
      int width = -1;
      int height = -1;
      int channels = -1;

      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gasenin_l_image_smooth, "pic.jpg");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 1);

      if (data == nullptr) {
        width = 64;
        height = 64;
        input_data_.width = width;
        input_data_.height = height;
        input_data_.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        for (int i = 0; i < width * height; ++i) {
          input_data_.data[i] = (i % 2) * 255;
        }
      } else {
        input_data_.width = width;
        input_data_.height = height;
        input_data_.data =
            std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width) * static_cast<ptrdiff_t>(height)));
        stbi_image_free(data);
      }
    }

    input_data_.kernel_size = kernel_size;

    ref_output_ = input_data_;
    ref_output_.data.assign(input_data_.data.size(), 0);

    GaseninLImageSmoothSEQ task(input_data_);

    task.Validation();
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    auto result_task = task.GetOutput();
    ref_output_.data = result_task.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    return output_data == ref_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_;
};

namespace {

TEST_P(GaseninLRunFuncTestsProcesses, ImageSmoothing) {
  ExecuteTest(GetParam());
}

TEST(Gasenin_L_Image_Smooth_Common, TaskData_Coverage_Ultimate) {
  gasenin_l_image_smooth::TaskData d1;
  gasenin_l_image_smooth::TaskData d2;

  EXPECT_TRUE(d1 == d2);
  EXPECT_FALSE(d1.operator!=(d2));

  d2.width = 10;
  EXPECT_FALSE(d1 == d2);
  EXPECT_TRUE(d1.operator!=(d2));

  d2 = d1;
  d2.height = 10;
  EXPECT_TRUE(d1.operator!=(d2));

  d2 = d1;
  d2.kernel_size = 10;
  EXPECT_TRUE(d1.operator!=(d2));

  d2 = d1;
  d2.data = {1, 2, 3};
  EXPECT_TRUE(d1.operator!=(d2));
}

TEST(Gasenin_L_Image_Smooth_SEQ, RunImpl_Coverage_Fix) {
  gasenin_l_image_smooth::TaskData in;
  in.width = 1;
  in.height = 1;
  in.kernel_size = 1;
  in.data = {100};

  gasenin_l_image_smooth::OutType out;
  out.data.resize(1);

  auto task = std::make_shared<gasenin_l_image_smooth::GaseninLImageSmoothSEQ>(in);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  EXPECT_EQ(task->GetOutput().data[0], 100);
}

TEST(Gasenin_L_Image_Smooth_MPI, TaskData_Coverage_Test) {
  gasenin_l_image_smooth::TaskData d1;
  gasenin_l_image_smooth::TaskData d2;

  ASSERT_TRUE(d1 == d2);

  d2.width = 5;
  ASSERT_TRUE(d1.operator!=(d2));
  ASSERT_FALSE(d1 == d2);
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(3, "kernel3"), std::make_tuple(5, "kernel5"),
                                            std::make_tuple(7, "kernel7"), std::make_tuple(3, "small_image"),
                                            std::make_tuple(3, "tiny_image_for_coverage")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaseninLImageSmoothMPI, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth),
    ppc::util::AddFuncTask<GaseninLImageSmoothSEQ, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaseninLRunFuncTestsProcesses::PrintFuncTestName<GaseninLRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothTests, GaseninLRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_image_smooth
