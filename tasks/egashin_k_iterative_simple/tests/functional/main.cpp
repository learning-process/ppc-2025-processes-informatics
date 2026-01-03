#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "egashin_k_iterative_simple/mpi/include/ops_mpi.hpp"
#include "egashin_k_iterative_simple/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_iterative_simple {

class EgashinKRunFuncTestsIterativeSimple : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    if (output_data.size() != expected_data_.size()) {
      return false;
    }

    double tolerance = input_data_.tolerance;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_data_[i]) > tolerance * 10) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  // если убрать {}, жалуется valgrind, если оставить clang-tidy, поэтому nolint
  OutType expected_data_{};  // NOLINT(readability-redundant-member-init)
};

namespace {

TestType CreateTestCase(const std::vector<std::vector<double>> &matrix, const std::vector<double> &b,
                        const std::vector<double> &x0, double tolerance, int max_iter,
                        const std::vector<double> &expected, const std::string &name) {
  InType input;
  input.A = matrix;
  input.b = b;
  input.x0 = x0;
  input.tolerance = tolerance;
  input.max_iterations = max_iter;
  return std::make_tuple(input, expected, name);
}

TEST_P(EgashinKRunFuncTestsIterativeSimple, IterativeMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    CreateTestCase({{2.0, 1.0}, {1.0, 2.0}}, {3.0, 3.0}, {0.0, 0.0}, 1e-6, 1000, {1.0, 1.0}, "Diag2x2Basic"),
    CreateTestCase({{4.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 4.0}}, {5.0, 6.0, 5.0}, {0.0, 0.0, 0.0}, 1e-6, 1000,
                   {1.0, 1.0, 1.0}, "Tridiag3x3"),
    CreateTestCase({{3.0, 1.0}, {1.0, 3.0}}, {4.0, 4.0}, {0.0, 0.0}, 1e-6, 1000, {1.0, 1.0}, "Diag2x2Sym"),
    CreateTestCase({{5.0, 2.0}, {2.0, 5.0}}, {7.0, 7.0}, {0.0, 0.0}, 1e-6, 1000, {1.0, 1.0}, "Diag2x2Strong"),
    CreateTestCase({{10.0, 1.0}, {1.0, 10.0}}, {11.0, 11.0}, {0.0, 0.0}, 1e-6, 1000, {1.0, 1.0}, "Diag2x2Dominant")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EgashinKIterativeSimpleSEQ, InType>(kTestParam, PPC_SETTINGS_egashin_k_iterative_simple),
    ppc::util::AddFuncTask<EgashinKIterativeSimpleMPI, InType>(kTestParam, PPC_SETTINGS_egashin_k_iterative_simple));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = EgashinKRunFuncTestsIterativeSimple::PrintFuncTestName<EgashinKRunFuncTestsIterativeSimple>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(IterativeMethodTests, EgashinKRunFuncTestsIterativeSimple, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace egashin_k_iterative_simple
