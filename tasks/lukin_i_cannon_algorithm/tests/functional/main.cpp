#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lukin_i_cannon_algorithm {
const double EPSILON = 1e-9;

class LukinIRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lukin_i_cannon_algorithm, params + ".txt");

    std::ifstream ifstr(path);

    size = 0;
    ifstr >> size;

    std::vector<double> A(size * size);
    std::vector<double> B(size * size);
    std::vector<double> C(size * size, 0);

    for (int i = 0; i < size * size; i++) {
      ifstr >> A[i];
    }

    for (int i = 0; i < size * size; i++) {
      ifstr >> B[i];
    }

    input_data_ = std::make_tuple(A, B, size);

    for (int i = 0; i < size; i++) {
      for (int k = 0; k < size; k++) {
        double fixed = A[i * size + k];
        for (int j = 0; j < size; j++) {
          C[i * size + j] += fixed * B[k * size + j];
        }
      }
    }

    expected_data_ = C;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (int i = 0; i < expected_data_.size(); i++) {
      if (std::abs(expected_data_[i] - output_data[i]) > EPSILON) {
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

  OutType expected_data_{};

  int size = 0;
};

namespace {

TEST_P(LukinIRunFuncTestsProcesses, Cannon) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    "matrix_2",
    "matrix_4",
    "matrix_8",
    "matrix_12",
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LukinICannonAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_cannon_algorithm),
    ppc::util::AddFuncTask<LukinICannonAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_cannon_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses::PrintFuncTestName<LukinIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(CannonAlgorithmTest, LukinIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_cannon_algorithm
