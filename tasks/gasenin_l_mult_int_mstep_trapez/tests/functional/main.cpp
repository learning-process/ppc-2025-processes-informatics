#include <gtest/gtest.h>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"
#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLMultIntMstepTrapezRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &data = std::get<0>(test_param);
    std::string desc = std::get<1>(test_param);
    return desc + "_steps" + std::to_string(data.n_steps) + "_func" + std::to_string(data.func_id);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    exact_integral_ = GetExactIntegral(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для функций 0-2 сравниваем с точным значением
    if (input_data_.func_id >= 0 && input_data_.func_id <= 2) {
      // ДОПУСК 2% - метод трапеций дает такую ошибку при малом n
      double tolerance = 2e-2 * std::abs(exact_integral_) + 1e-4;

      return std::abs(output_data - exact_integral_) < tolerance;
    } else {
      // Для функций без точного интеграла проверяем корректность
      return std::isfinite(output_data);
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double exact_integral_ = 0.0;
};

namespace {

TEST_P(GaseninLMultIntMstepTrapezRunFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

// ОБНОВЛЕННЫЙ набор тестов с бОльшим количеством шагов
const std::array<TestType, 8> kTestParam = {
    // Простые случаи для функции 0: x+y
    std::make_tuple(TaskData{100, 0, 0.0, 1.0, 0.0, 1.0}, "simple_unit_square_func0"),
    std::make_tuple(TaskData{200, 0, 0.0, 2.0, 0.0, 3.0}, "rectangle_2x3_func0"),

    // Функция 1: x^2 + y^2 - увеличиваем шаги для точности
    std::make_tuple(TaskData{150, 1, 0.0, 1.0, 0.0, 1.0}, "unit_square_func1"),
    std::make_tuple(TaskData{200, 1, 0.0, 2.0, 0.0, 1.0}, "rectangle_2x1_func1"),

    // Функция 2: sin(x)*cos(y) - УВЕЛИЧИВАЕМ шаги до 100+ для точности
    std::make_tuple(TaskData{120, 2, 0.0, 3.1415926535, 0.0, 3.1415926535}, "pi_square_func2"),
    std::make_tuple(TaskData{100, 2, 0.0, 1.5707963268, 0.0, 1.5707963268}, "half_pi_square_func2"),

    // Функции без аналитического решения
    std::make_tuple(TaskData{100, 3, 0.0, 1.0, 0.0, 1.0}, "exp_func3"),
    std::make_tuple(TaskData{100, 4, 0.0, 1.0, 0.0, 1.0}, "sqrt_func4"),
};

// Создаем задачи для MPI и SEQ версий
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez),
                                           ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    GaseninLMultIntMstepTrapezRunFuncTests::PrintFuncTestName<GaseninLMultIntMstepTrapezRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegrationTests, GaseninLMultIntMstepTrapezRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_mult_int_mstep_trapez
