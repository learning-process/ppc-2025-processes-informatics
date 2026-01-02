#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"
#include "dorofeev_i_ccs_martrix_production/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_ccs_martrix_production/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dorofeev_i_ccs_martrix_production {

class DorofeevICCSMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<1>(param);
  }

 protected:
  void SetUp() override {
    /*
      A = [ 1 0 ]
          [ 0 2 ]

      B = [ 3 0 ]
          [ 0 4 ]

      C = A * B = [ 3 0 ]
                    [ 0 8 ]
    */

    CCSMatrix a;
    a.rows = 2;
    a.cols = 2;
    a.col_ptr = {0, 1, 2};
    a.row_indices = {0, 1};
    a.values = {1.0, 2.0};

    CCSMatrix b;
    b.rows = 2;
    b.cols = 2;
    b.col_ptr = {0, 1, 2};
    b.row_indices = {0, 1};
    b.values = {3.0, 4.0};

    input_ = std::make_pair(a, b);

    expected_.rows = 2;
    expected_.cols = 2;
    expected_.col_ptr = {0, 1, 2};
    expected_.row_indices = {0, 1};
    expected_.values = {3.0, 8.0};
  }

  bool CheckTestOutputData(OutType &out) final {
    if (out.rows != expected_.rows || out.cols != expected_.cols) {
      return false;
    }

    if (out.col_ptr != expected_.col_ptr) {
      return false;
    }

    if (out.row_indices != expected_.row_indices) {
      return false;
    }

    const double eps = 1e-9;
    for (size_t i = 0; i < out.values.size(); i++) {
      if (std::abs(out.values[i] - expected_.values[i]) > eps) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

TEST_P(DorofeevICCSMatrixFuncTests, CCSMatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParams = {
    std::make_tuple(0, "basic_ccs_mul"),
};

const auto kTaskList = std::tuple_cat(ppc::util::AddFuncTask<DorofeevICCSMatrixProductionMPI, InType>(
                                          kTestParams, PPC_SETTINGS_dorofeev_i_ccs_martrix_production),
                                      ppc::util::AddFuncTask<DorofeevICCSMatrixProductionSEQ, InType>(
                                          kTestParams, PPC_SETTINGS_dorofeev_i_ccs_martrix_production));

INSTANTIATE_TEST_SUITE_P(CCSMatrixTests, DorofeevICCSMatrixFuncTests, ppc::util::ExpandToValues(kTaskList),
                         DorofeevICCSMatrixFuncTests::PrintFuncTestName<DorofeevICCSMatrixFuncTests>);

}  // namespace dorofeev_i_ccs_martrix_production
