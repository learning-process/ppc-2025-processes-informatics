#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

struct ExpectedSolution {
  std::vector<double> argmins;
  double value;
};

struct LocalTestCase {
  std::string name;
  std::string function_id;
  Problem problem;
  ExpectedSolution expected;
};

using FuncParam = ppc::util::FuncTestParam<InType, OutType, LocalTestCase>;

namespace {

using Func = Function;

const std::unordered_map<std::string, Func> kFunctionRegistry = {
    {"linear_inc", [](double x) { return x; }},
    {"linear_dec", [](double x) { return -2.0 * x + 3.0; }},
    {"quad", [](double x) { return x * x; }},
    {"quad_shift_1", [](double x) { return (x - 1.0) * (x - 1.0); }},
    {"quad_plus_5", [](double x) { return x * x + 5.0; }},
    {"abs", [](double x) { return std::abs(x); }},
    {"abs_shift_1", [](double x) { return std::abs(x - 1.0); }},
    {"abs_plus_quad", [](double x) { return std::abs(x) + 0.1 * x * x; }},
    {"quad_mixed", [](double x) { return 0.2 * x * x - 0.3 * x; }},
    {"quartic", [](double x) { return x * x * x * x; }},
    {"exp_quad", [](double x) { return std::exp(x * x); }},
    {"log_quad", [](double x) { return std::log(x * x + 2.0); }},
};

}  // namespace

static std::vector<LocalTestCase> LoadTestCasesFromData() {
  const std::filesystem::path path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json data;
  file >> data;

  if (!data.is_array()) {
    throw std::runtime_error("tests.json must contain array");
  }

  std::vector<LocalTestCase> cases;
  cases.reserve(data.size());

  for (const auto &item : data) {
    LocalTestCase tc;

    tc.name = item.at("name").get<std::string>();
    tc.function_id = item.at("function").get<std::string>();

    const auto &pj = item.at("problem");
    tc.problem.left = pj.at("left");
    tc.problem.right = pj.at("right");
    tc.problem.accuracy = kDefaultAccuracy;
    tc.problem.reliability = kDefaultReliability;
    tc.problem.max_iterations = kDefaultMaxIterations;

    const auto it = kFunctionRegistry.find(tc.function_id);
    if (it == kFunctionRegistry.end()) {
      throw std::runtime_error("Unknown function id: " + tc.function_id);
    }
    tc.problem.func = it->second;

    const auto &ej = item.at("expected");
    tc.expected.argmins = ej.at("argmins").get<std::vector<double>>();
    tc.expected.value = ej.at("value");

    cases.push_back(std::move(tc));
  }

  return cases;
}

static std::vector<FuncParam> BuildTestTasks(const std::vector<LocalTestCase> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2);

  const std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  const std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

class SizovDRunFuncTestsGlobalSearch : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestCase> {
 public:
  static std::string PrintTestParam(const LocalTestCase &tc) {
    return tc.name;
  }

 protected:
  void SetUp() override {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = test_case_.problem;
    expected_ = test_case_.expected;
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    if (!std::isfinite(out.value)) {
      return false;
    }

    if (std::abs(out.value - expected_.value) > 20.0 * input_.accuracy) {
      return false;
    }

    double best_dx = std::numeric_limits<double>::infinity();
    for (double a : expected_.argmins) {
      best_dx = std::min(best_dx, std::abs(out.argmin - a));
    }

    return best_dx <= 5.0 * input_.accuracy;
  }

 private:
  LocalTestCase test_case_;
  InType input_;
  ExpectedSolution expected_;
};

namespace {

TEST_P(SizovDRunFuncTestsGlobalSearch, FromJson) {
  ExecuteTest(GetParam());
}

const auto kCases = LoadTestCasesFromData();
const auto kTasks = BuildTestTasks(kCases);
const auto kValues = ::testing::ValuesIn(kTasks);

INSTANTIATE_TEST_SUITE_P(SizovDGlobalSearch, SizovDRunFuncTestsGlobalSearch, kValues,
                         SizovDRunFuncTestsGlobalSearch::PrintFuncTestName<SizovDRunFuncTestsGlobalSearch>);

}  // namespace

}  // namespace sizov_d_global_search
