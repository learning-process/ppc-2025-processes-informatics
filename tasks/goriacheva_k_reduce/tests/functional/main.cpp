#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "goriacheva_k_reduce/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_reduce/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace goriacheva_k_reduce {

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class GoriachevaKReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<FuncParam> &info) {
    const auto &tc = std::get<2>(info.param);
    const auto &name = std::get<2>(tc);
    return name + "_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<2>(GetParam());
    input_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output) final {
    return output == expected_;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

static std::vector<FuncParam> LoadTests() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_goriacheva_k_reduce, "tests.json");

  std::ifstream fin(path);
  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> params;

  const std::string settings = PPC_SETTINGS_goriacheva_k_reduce;
  const std::string mpi_suffix = ppc::task::GetStringTaskType(GoriachevaKReduceMPI::GetStaticTypeOfTask(), settings);
  const std::string seq_suffix = ppc::task::GetStringTaskType(GoriachevaKReduceSEQ::GetStaticTypeOfTask(), settings);

  for (const auto &item : j) {
    TestType tc{item.at("input").get<InType>(), item.at("result").get<OutType>(), item.at("name").get<std::string>()};

    params.emplace_back(ppc::task::TaskGetter<GoriachevaKReduceMPI, InType>, std::get<2>(tc) + "_" + mpi_suffix, tc);

    params.emplace_back(ppc::task::TaskGetter<GoriachevaKReduceSEQ, InType>, std::get<2>(tc) + "_" + seq_suffix, tc);
  }

  return params;
}

const auto kParams = LoadTests();

TEST_P(GoriachevaKReduceFuncTests, ReduceSum) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKReduceFunctionalTests, GoriachevaKReduceFuncTests, testing::ValuesIn(kParams),
                         GoriachevaKReduceFuncTests::PrintTestParam);

}  // namespace goriacheva_k_reduce