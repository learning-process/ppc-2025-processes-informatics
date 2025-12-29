#include <gtest/gtest.h>
//#include <stb/stb_image.h>

//#include <algorithm>
//#include <array>
//#include <cstddef>
//#include <cstdint>
//#include <numeric>
//#include <stdexcept>
#include <string>
//#include <tuple>
//#include <utility>
#include <vector>
#include <fstream>
#include<nlohmann/json.hpp>
//#include<functional>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_violation_order_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace goriacheva_k_violation_order_elem_vec {

  using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class GoriachevaKViolationOrderElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<FuncParam> &info) {
    return std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input = std::get<0>(params);
    expected = std::get<1>(params);
  }

  InType GetTestInputData() final { return input; }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected;
  }

 private:
  InType input;
  OutType expected{};
};

namespace {

std::vector<FuncParam> LoadTestParams() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_goriacheva_k_violation_order_elem_vec, "tests.json");
  std::ifstream fin(path);
  if (!fin.is_open()) throw std::runtime_error("Cannot open tests.json");

  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> cases;
  cases.reserve(j.size() * 2);

  const std::string settings_path = PPC_SETTINGS_goriacheva_k_violation_order_elem_vec;
  const std::string mpi_suffix = ppc::task::GetStringTaskType(GoriachevaKViolationOrderElemVecMPI::GetStaticTypeOfTask(), settings_path);
  const std::string seq_suffix = ppc::task::GetStringTaskType(GoriachevaKViolationOrderElemVecSEQ::GetStaticTypeOfTask(), settings_path);

  for (const auto &item : j) {
    TestType tc{ item.at("input").get<InType>(), item.at("result").get<OutType>(), item.at("name").get<std::string>() };

    std::string mpi_name = std::get<2>(tc) + "_" + mpi_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKViolationOrderElemVecMPI, InType>, mpi_name, tc);

    std::string seq_name = std::get<2>(tc) + "_" + seq_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKViolationOrderElemVecSEQ, InType>, seq_name, tc);
  }

  return cases;
}

const std::vector<FuncParam> kFuncParams = LoadTestParams();

TEST_P(GoriachevaKViolationOrderElemVecFuncTests, VectorOrderViolations) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(FunctionalTests, GoriachevaKViolationOrderElemVecFuncTests, testing::ValuesIn(kFuncParams),
                         GoriachevaKViolationOrderElemVecFuncTests::PrintTestParam);

}  // namespace

}  // namespace goriacheva_k_violation_order_elem_vec