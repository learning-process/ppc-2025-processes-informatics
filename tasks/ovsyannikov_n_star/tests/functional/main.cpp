#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "ovsyannikov_n_star/common/include/common.hpp"
#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ovsyannikov_n_star {

class OvsyannikovNRunFuncTestsStar : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<TestType>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = input_data_[2];
  }

  bool CheckTestOutputData(OutType &actual_res) final {
    return actual_res == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_ = 0;
};

TEST_P(OvsyannikovNRunFuncTestsStar, TestRouting) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(std::vector<int>{1, 2, 42}, "From1To2"),
                                            std::make_tuple(std::vector<int>{0, 1, 100}, "From0To1"),
                                            std::make_tuple(std::vector<int>{2, 0, 77}, "From2To0")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OvsyannikovNStarMPI, InType>(kTestParam, PPC_SETTINGS_ovsyannikov_n_star),
                   ppc::util::AddFuncTask<OvsyannikovNStarSEQ, InType>(kTestParam, PPC_SETTINGS_ovsyannikov_n_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(StarTests, OvsyannikovNRunFuncTestsStar, kGtestValues,
                         OvsyannikovNRunFuncTestsStar::PrintFuncTestName<OvsyannikovNRunFuncTestsStar>);

}  // namespace ovsyannikov_n_star
