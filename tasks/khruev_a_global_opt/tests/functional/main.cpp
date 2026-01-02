#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"
#include "khruev_a_global_opt/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    // auto params = std::get<2>(test_param);
    std::string stroka = std::get<0>(test_param);
    return stroka;
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_.func_id = std::get<1>(param);
    input_data_.ax = std::get<2>(param);
    input_data_.bx = std::get<3>(param);
    input_data_.ay = std::get<4>(param);
    input_data_.by = std::get<5>(param);
    input_data_.epsilon = 0.0001;
    input_data_.max_iter = 5000;
    input_data_.r = 2.5;

    expected_min_ = std::get<6>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что найденное значение функции близко к теоретическому минимуму
    const double tolerance = 1e-2;
    return std::abs(output_data.value - expected_min_) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_min_;
};

namespace {

TEST_P(KhruevAGlobalOptFuncTests, CorrectMinimumFound) {
  ExecuteTest(GetParam());
}

// Определяем кейсы для тестов
// 1. Параболоид (ID=1) в [0, 1]x[0, 1]. Минимум в (0.5, 0.5) = 0.0
// 2. Растригин (ID=2) в [0, 1]x[0, 1]. Минимум в (0.5, 0.5) = 0.0 (так как мы сдвигали функцию)
// 3. Параболоид со смещенными границами [-0.5, 0.5].
//    У нас функция target_function(id=1) ожидает, что минимум в (0.5, 0.5) в локальных координатах.
//    Если мы хотим протестировать общие границы, нужно убедиться, что target_function это поддерживает.
//    В моей реализации target_function работает с реальными координатами.
//    Параболоид: (x-0.5)^2 + (y-0.5)^2. Минимум всегда в 0.5, 0.5.
//    Если границы [0, 1], точка 0.5 попадает. Если границы [0.6, 1.0], минимум будет на границе (0.6).

const std::array<TestType, 2> kTestCases = {std::make_tuple("Paraboloid", 1, 0.0, 1.0, 0.0, 1.0, 0.0),
                                            // std::make_tuple(1, 0.6, 1.0, 0.0, 1.0, 0.01),
                                            std::make_tuple("Rastrigin", 2, 0.0, 1.0, 0.0, 1.0, 0.0)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KhruevAGlobalOptMPI, InType>(kTestCases, PPC_SETTINGS_khruev_a_global_opt),
                   ppc::util::AddFuncTask<KhruevAGlobalOptSEQ, InType>(kTestCases, PPC_SETTINGS_khruev_a_global_opt));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KhruevAGlobalOptFuncTests::PrintFuncTestName<KhruevAGlobalOptFuncTests>;

INSTANTIATE_TEST_SUITE_P(GlobalOptTests, KhruevAGlobalOptFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace khruev_a_global_opt
