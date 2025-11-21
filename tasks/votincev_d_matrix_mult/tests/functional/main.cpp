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

    input_data_ = std::make_tuple(m, n, k, A, B);


    // вычисляю предполагаемый результат
    expected_res_.assign(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int t = 0; t < k; ++t) {
          sum += A[i * k + t] * B[t * n + j];
        }
        expected_res_[i * n + j] = sum;
      }
    }

    
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1,2... процессы не владеют нужным результатом
    if(output_data.size() != expected_res_.size()) {
      return true;
    }

    //std::cout << "\n\nProcess0 is checking result\n\n";
    // 0й процесс должен иметь корректную матрицу после умножения
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

const std::array<TestType, 10> kTestParam = {
    "test1", "test2", "test3", "test4", "test5", "test6","test7","test8","test9","test10"
};

// const std::array<TestType, 3> kTestParam = {
//     "test1","test2","test3"
// };

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
