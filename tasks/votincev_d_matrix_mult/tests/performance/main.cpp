#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"
#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"
#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultRunPerfTestsProcesses
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  InType GetTestInputData() final { return input_data_; }

 protected:
  InType input_data_;
  OutType expected_res_;

  void SetUp() override {
    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_votincev_d_matrix_mult, "testPerf.txt");

    std::ifstream file(file_path);
    if (!file.is_open()) {
      return;
    }

    int m, n, k;
    file >> m >> n >> k;

    std::vector<double> A(m * k);
    std::vector<double> B(k * n);

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
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VotincevDMatrixMultMPI, VotincevDMatrixMultSEQ>(
        PPC_SETTINGS_votincev_d_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = VotincevDMatrixMultRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDMatrixMultRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDMatrixMultRunPerfTestsProcesses,
                         kGtestValues, kPerfTestName);

}  // namespace votincev_d_matrix_mult
