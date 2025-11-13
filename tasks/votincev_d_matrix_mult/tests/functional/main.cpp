#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"
#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"
#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param =
        std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string input_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_votincev_d_matrix_mult, param + ".txt");

    std::ifstream file(input_path);
    if (!file.is_open()) return;

    int m, n, k;
    file >> m >> n >> k;

    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> R_expected(m * n);

    for (double &v : A) file >> v;
    for (double &v : B) file >> v;
    for (double &v : R_expected) file >> v;

    input_data_ = std::make_tuple(m, n, k, A, B);
    expected_res_ = R_expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_res_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_res_;
};

namespace {

TEST_P(VotincevDMatrixMultRunFuncTestsProcesses, MatrixMultiplicationTests) {
  ExecuteTest(GetParam());
}

// const std::array<TestType, 6> kTestParam = {
//     "test1", "test2", "test3", "test4", "test5", "test6"
// };

const std::array<TestType, 2> kTestParam = {
    "test1","test2"
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<VotincevDMatrixMultMPI, InType>(
                       kTestParam, PPC_SETTINGS_votincev_d_matrix_mult),
                   ppc::util::AddFuncTask<VotincevDMatrixMultSEQ, InType>(
                       kTestParam, PPC_SETTINGS_votincev_d_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VotincevDMatrixMultRunFuncTestsProcesses::PrintFuncTestName<
        VotincevDMatrixMultRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, VotincevDMatrixMultRunFuncTestsProcesses,
                         kGtestValues, kPerfTestName);

}  // namespace

}  // namespace votincev_d_matrix_mult
