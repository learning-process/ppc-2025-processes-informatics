#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"
#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_mult_int_trapez {

class FrolovaRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    // Получаем параметры теста
    unsigned int test_id = std::get<0>(params);
    std::string test_type = std::get<1>(params);

    // Создаем входные данные в зависимости от test_id
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    // Пример: создаем простые тестовые данные
    input_data_.limits.clear();
    input_data_.number_of_intervals.clear();

    // Разные размерности в зависимости от test_id
    unsigned int dim = (test_id % 3) + 1;  // 1, 2 или 3 измерения

    for (unsigned int i = 0; i < dim; ++i) {
      double a = dist(gen);
      double b = a + dist(gen) + 1.0;  // гарантируем b > a
      input_data_.limits.emplace_back(a, b);
      input_data_.number_of_intervals.push_back(50 + (test_id * 10) % 50);
    }

    // Задаем функцию в зависимости от test_type
    if (test_type == "poly") {
      input_data_.function = [](std::vector<double> input) {
        double result = 1.0;
        for (double val : input) {
          result *= val;
        }
        return result;
      };
    } else if (test_type == "sin") {
      input_data_.function = [](std::vector<double> input) {
        double result = 0.0;
        for (double val : input) {
          result += std::sin(val);
        }
        return result;
      };
    } else {
      // Функция по умолчанию
      input_data_.function = [](std::vector<double> input) {
        double result = 0.0;
        for (double val : input) {
          result += val;
        }
        return result;
      };
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что результат вычислен (не NaN и не бесконечность)
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(FrolovaRunFuncTestsProcesses, TrapezoidalIntegration) {
  ExecuteTest(GetParam());
}

// Параметры тестов: (id, тип)
const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "poly"), std::make_tuple(2, "sin"),
                                            std::make_tuple(3, "poly"), std::make_tuple(4, "sin"),
                                            std::make_tuple(5, "poly"), std::make_tuple(6, "sin")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez),
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezSEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FrolovaRunFuncTestsProcesses::PrintFuncTestName<FrolovaRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TrapezoidalIntegrationTests, FrolovaRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_mult_int_trapez
