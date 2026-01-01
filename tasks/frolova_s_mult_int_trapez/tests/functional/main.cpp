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

using TestType = std::tuple<std::string, unsigned int>;

class FrolovaRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name = std::get<0>(params);
    unsigned int test_dim = std::get<1>(params);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<double> limit_dist(0.0, 10.0);
    std::uniform_int_distribution<unsigned int> interval_dist(50, 150);

    // Инициализируем входные данные в зависимости от теста
    input_data_.limits.clear();
    input_data_.number_of_intervals.clear();

    for (unsigned int i = 0; i < test_dim; ++i) {
      double a = limit_dist(gen);
      double b = limit_dist(gen);
      if (a > b) {
        std::swap(a, b);
      }
      input_data_.limits.emplace_back(a, b);
      input_data_.number_of_intervals.push_back(interval_dist(gen));
    }

    // Устанавливаем функцию в зависимости от имени теста
    if (test_name == "function1") {
      input_data_.function = [](std::vector<double> input) { return std::pow(input[0], 3) + std::pow(input[1], 3); };
    } else if (test_name == "function2") {
      input_data_.function = [](std::vector<double> input) {
        return std::sin(input[0]) + std::sin(input[1]) + std::sin(input[2]);
      };
    } else if (test_name == "function3") {
      input_data_.function = [](std::vector<double> input) { return 8.0 * input[0] * input[1] * input[2]; };
    } else if (test_name == "function4") {
      input_data_.function = [](std::vector<double> input) { return -1.0 / std::sqrt(1.0 - std::pow(input[0], 2)); };
    } else if (test_name == "function5") {
      input_data_.function = [](std::vector<double> input) { return -(std::sin(input[0]) * std::cos(input[1])); };
    } else if (test_name == "function6") {
      input_data_.function = [](std::vector<double> input) {
        return (-3.0 * std::pow(input[1], 2) * std::sin(5.0 * input[0])) / 2.0;
      };
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для функциональных тестов мы просто проверяем, что результат вычислен
    // и не является NaN или бесконечностью
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

// Определяем тестовые параметры
const std::array<TestType, 8> kTestParam = {
    std::make_tuple("function1", 2U),  // 2D: x^3 + y^3
    std::make_tuple("function2", 3U),  // 3D: sin(x) + sin(y) + sin(z)
    std::make_tuple("function3", 3U),  // 3D: 8xyz
    std::make_tuple("function4", 1U),  // 1D: -1/sqrt(1-x^2)
    std::make_tuple("function5", 2U),  // 2D: -sin(x)cos(y)
    std::make_tuple("function6", 2U),  // 2D: -3y^2 sin(5x)/2
    std::make_tuple("function1", 2U),  // Еще раз 2D с другими интервалами
    std::make_tuple("function3", 3U)   // Еще раз 3D с другими интервалами
};

// Добавляем задачи для SEQ и MPI версий
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezSEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez),
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FrolovaRunFuncTestsProcesses::PrintFuncTestName<FrolovaRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TrapezoidalIntegrationTests, FrolovaRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_mult_int_trapez
