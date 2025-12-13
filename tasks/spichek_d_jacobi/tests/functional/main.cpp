#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_jacobi {

// Функция вычисления точного решения метода Якоби (для тестов)
Vector CalculateExpectedJacobiSolution(const Matrix &A, const Vector &b, double tolerance, int max_iter_limit) {
  size_t n = A.size();
  if (n == 0) {
    return Vector{};
  }

  Vector x_k(n, 0.0);
  Vector x_k_plus_1(n, 0.0);
  double max_diff;
  int iter = 0;

  do {
    iter++;

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      double a_ii = A[i][i];
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i][j] * x_k[j];
        }
      }
      x_k_plus_1[i] = (b[i] - sum) / a_ii;
    }

    // Используем максимум по компонентам для сходимости (∞-норма)
    max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = std::abs(x_k_plus_1[i] - x_k[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }

    x_k = x_k_plus_1;

  } while (max_diff > tolerance && iter < max_iter_limit);

  return x_k_plus_1;
}

// Тестовый класс для запуска функциональных тестов
class SpichekDJacobiRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[in_data, description] = test_param;
    return description;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[A, b, eps_test, max_iter_test] = input_data_;
    size_t n = A.size();
    if (n == 0) {
      return output_data.empty();
    }

    constexpr double kExpectedTolerance = 1e-12;
    constexpr int kExpectedMaxIter = 500;

    Vector expected_result = CalculateExpectedJacobiSolution(A, b, kExpectedTolerance, kExpectedMaxIter);

    // Сравниваем по ∞-норме
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = std::abs(output_data[i] - expected_result[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }

    constexpr double kComparisonTolerance = 1e-6;
    return (max_diff < kComparisonTolerance);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SpichekDJacobiRunFuncTestsProcesses, JacobiTest) {
  ExecuteTest(GetParam());
}

constexpr double kDefaultEps = 1e-4;
constexpr int kDefaultMaxIter = 100;
constexpr int kHighMaxIter = 500;

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(std::make_tuple(Matrix{{4.0, -1.0}, {-1.0, 3.0}}, Vector{2.0, 5.0}, kDefaultEps, kDefaultMaxIter),
                    "basic_2x2_diagonally_dominant"),
    std::make_tuple(std::make_tuple(Matrix{{10.0, -1.0, 2.0}, {-1.0, 11.0, -1.0}, {2.0, -1.0, 10.0}},
                                    Vector{7.0, 11.0, 17.0}, kDefaultEps, kDefaultMaxIter),
                    "3x3_converging"),
    std::make_tuple(
        std::make_tuple(
            Matrix{{100.0, 1.0, 2.0, 3.0}, {1.0, 200.0, 3.0, 4.0}, {2.0, 3.0, 300.0, 4.0}, {3.0, 4.0, 4.0, 400.0}},
            Vector{100.0, 200.0, 300.0, 400.0}, 1e-2, kDefaultMaxIter),
        "4x4_large_coeffs_low_eps"),
    std::make_tuple(
        std::make_tuple(Matrix{{3.0, 1.0, 1.0}, {1.0, 3.0, 1.0}, {1.0, 1.0, 3.0}}, Vector{5.0, 5.0, 5.0}, 1e-6, 10),
        "slow_convergence_max_iter_limit"),
    std::make_tuple(std::make_tuple(Matrix{{3.0, 1.0, 1.0}, {1.0, 3.0, 1.0}, {1.0, 1.0, 3.0}}, Vector{5.0, 5.0, 5.0},
                                    1e-6, kHighMaxIter),
                    "high_eps_high_max_iter"),
    std::make_tuple(std::make_tuple(Matrix{}, Vector{}, kDefaultEps, kDefaultMaxIter), "empty_system")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SpichekDJacobiMPI, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi),
                   ppc::util::AddFuncTask<SpichekDJacobiSEQ, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = SpichekDJacobiRunFuncTestsProcesses::PrintFuncTestName<SpichekDJacobiRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(JacobiTests, SpichekDJacobiRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace spichek_d_jacobi
