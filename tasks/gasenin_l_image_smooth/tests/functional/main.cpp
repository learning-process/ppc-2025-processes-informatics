#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <string>
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
    int width = -1;
    int height = -1;
    int channels = -1;

    // Загрузка изображения
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gasenin_l_image_smooth, "pic.jpg");
    // Грузим как GreyScale (1 канал) для упрощения задачи сглаживания в лабе
    auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 1);

    if (data == nullptr) {
      // Если картинки нет, создаем синтетическую "шахматную доску" для теста
      width = 64;
      height = 64;
      input_data_.width = width;
      input_data_.height = height;
      input_data_.data.resize(width * height);
      for (int i = 0; i < width * height; ++i) {
        input_data_.data[i] = (i % 2) * 255;
      }
    } else {
      input_data_.width = width;
      input_data_.height = height;
      input_data_.data = std::vector<uint8_t>(data, data + (width * height));
      stbi_image_free(data);
    }

    // ИСПРАВЛЕНИЕ 1: Правильное извлечение параметров теста из кортежа GTest
    auto test_params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_.kernel_size = std::get<0>(test_params);

    // Генерируем reference result (последовательная версия)
    ref_output_ = input_data_;  // копируем метаданные
    ref_output_.data.assign(input_data_.data.size(), 0);

    // Простой последовательный прогон для проверки (Reference)
    GaseninLImageSmoothSEQ task(input_data_);

    // ИСПРАВЛЕНИЕ 2: Вызов публичных методов интерфейса, а не приватных Impl
    task.Validation();
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    // Копируем результат последовательной задачи в reference
    auto result_task = task.GetOutput();
    ref_output_.data = result_task.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      return output_data == ref_output_;
    }
    return true;  // Остальные процессы всегда возвращают true, чтобы не валить тест
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

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "kernel3"), std::make_tuple(5, "kernel5"),
                                            std::make_tuple(7, "kernel7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaseninLImageSmoothMPI, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth),
    ppc::util::AddFuncTask<GaseninLImageSmoothSEQ, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaseninLRunFuncTestsProcesses::PrintFuncTestName<GaseninLRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothTests, GaseninLRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_image_smooth
