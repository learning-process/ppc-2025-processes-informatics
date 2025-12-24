#include <gtest/gtest.h>

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"
#include "romanova_v_jacobi_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanova_v_jacobi_method {

class RomanovaVJacobiMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    TestType path = "perfTest";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_romanova_v_jacobi_method, path);
    std::ifstream file(abs_path + ".txt");
    if (file.is_open()) {
      int rows = 0;
      int columns = 0;
      size_t iterations = 0;
      double eps = 0.0;
      file >> rows >> columns >> iterations >> eps;
      exp_answer_ = OutType(rows);
      for (int i = 0; i < rows; i++) {
        file >> exp_answer_[i];
      }

      std::vector<double> x(rows);
      for (int i = 0; i < rows; i++) {
        file >> x[i];
      }

      std::vector<double> b(rows);
      for (int i = 0; i < rows; i++) {
        file >> b[i];
      }

      std::vector<std::vector<double>> A(rows, std::vector<double>(columns));
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
          file >> A[i][j];
        }
      }

      input_data_ = std::make_tuple(x, A, b, eps, iterations);
      eps_ = eps;

      file.close();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != exp_answer_.size()) {
      return false;
    }
    for (int i = 0; i < output_data.size(); i++) {
      if (abs(output_data[i] - exp_answer_[i]) > eps_) {
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
  OutType exp_answer_{};
  double eps_ = 0.0;
};

TEST_P(RomanovaVJacobiMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RomanovaVJacobiMethodMPI, RomanovaVJacobiMethodSEQ>(
    PPC_SETTINGS_romanova_v_jacobi_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovaVJacobiMethodPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovaVJacobiMethodPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace romanova_v_jacobi_method
