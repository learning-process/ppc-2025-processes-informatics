#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "lifanov_k_adj_inv_count/common/include/common.hpp"
#include "lifanov_k_adj_inv_count/mpi/include/ops_mpi.hpp"
#include "lifanov_k_adj_inv_count/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lifanov_k_adj_inv_count {

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class LifanovKRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const ::testing::TestParamInfo<FuncParam> &info) {
    return std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return data_;
  }

 private:
  OutType expected_{0};
  std::string filename_;
  InType data_;
};

namespace {

TEST_P(LifanovKRunFuncTests, AdjacentInversionCount) {
  ExecuteTest(GetParam());
}

std::vector<FuncParam> LoadTestParams() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lifanov_k_adj_inv_count, "tests.json");

  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> cases;
  cases.reserve(j.size() * 2);

  for (const auto &item : j) {
    TestType test_case{item.at("input").get<InType>(), item.at("expected").get<OutType>(),
                       item.at("name").get<std::string>()};

    const auto &task_name = std::get<2>(test_case);

    cases.emplace_back(ppc::task::TaskGetter<LifanovKAdjacentInversionCountMPI, InType>, task_name + "_mpi", test_case);

    cases.emplace_back(ppc::task::TaskGetter<LifanovKAdjacentInversionCountSEQ, InType>, task_name + "_seq", test_case);
  }

  return cases;
}

const std::vector<FuncParam> &GetTestParams() {
  static const std::vector<FuncParam> kParams = LoadTestParams();
  return kParams;
}

INSTANTIATE_TEST_SUITE_P(FunctionalTests, LifanovKRunFuncTests, ::testing::ValuesIn(GetTestParams()),
                         LifanovKRunFuncTests::PrintTestParam);

}  // namespace

}  // namespace lifanov_k_adj_inv_count
