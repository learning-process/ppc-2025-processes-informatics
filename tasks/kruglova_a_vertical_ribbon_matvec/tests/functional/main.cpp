#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"
#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "x" + std::to_string(std::get<1>(param));
  }

 protected:
  void SetUp() override {
    // 1. Извлекаем параметры (размеры матрицы) из параметров теста GTest
    // Исправлена синтаксическая ошибка в static_cast
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const int rows = std::get<0>(params);
    const int cols = std::get<1>(params);

    // 2. Выделяем память под матрицу и векторы
    std::vector<double> matrix(rows * cols);
    std::vector<double> vec(cols);

    // 3. Линейное заполнение данных (1, 2, 3...)
    // Это удобно для отладки, так как результат предсказуем
    for (int i = 0; i < rows * cols; ++i) {
      matrix[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < cols; ++i) {
      vec[i] = static_cast<double>(i + 1);
    }

    // 4. Вычисляем эталонный результат (последовательное умножение)
    // Это то, с чем будет сравниваться результат MPI
    ref_output_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < cols; ++j) {
        // Доступ к элементу: row * width + col
        sum += matrix[i * cols + j] * vec[j];
      }
      ref_output_[i] = sum;
    }

    // 5. Сохраняем входные данные для задачи
    // Порядок в InType: <rows, cols, matrix, vector>
    input_data_ = std::make_tuple(rows, cols, std::move(matrix), std::move(vec));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ВАЖНО: Проверяем результат только на главном процессе!
    // Остальные процессы возвращают true, чтобы не ломать тест.
    if (rank != 0) {
      return true;
    }

    if (output_data.size() != ref_output_.size()) {
      return false;
    }

    const double kEpsilon = 1e-6;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - ref_output_[i]) > kEpsilon) {
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
  OutType ref_output_;
};

namespace {

TEST_P(KruglovaAVerticalRibMatFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(10, 10),    // Малая квадратная
                                            std::make_tuple(100, 100),  // Средняя квадратная
                                            std::make_tuple(500, 100),  // Вертикальная (много строк)
                                            std::make_tuple(100, 500),  // Горизонтальная (много столбцов)
                                            std::make_tuple(50, 50)};   // Дополнительный тест;

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KruglovaAVerticalRibbMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec),
                                           ppc::util::AddFuncTask<KruglovaAVerticalRibbMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KruglovaAVerticalRibMatFuncTests::PrintFuncTestName<KruglovaAVerticalRibMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KruglovaAVerticalRibMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kruglova_a_vertical_ribbon_matvec
