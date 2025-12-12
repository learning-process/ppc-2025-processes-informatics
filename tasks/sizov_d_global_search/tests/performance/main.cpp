#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "performance/include/performance.hpp"
#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

constexpr double kLeft = -5.0;
constexpr double kRight = 5.0;
constexpr double kAccuracy = 1e-4;
constexpr double kReliability = 3.0;
constexpr int kMaxIterations = 1'000;

static InType MakePerfProblem() {
  InType p{};
  p.left = kLeft;
  p.right = kRight;
  p.accuracy = kAccuracy;
  p.reliability = kReliability;
  p.max_iterations = kMaxIterations;

  p.func = [](double x) {
    double v = 0.002 * x * x;
    v += 5.0 * std::sin(30.0 * x);
    v += std::sin(200.0 * std::sin(50.0 * x));
    v += 0.1 * std::cos(300.0 * x);
    return v;
  };

  return p;
}

template <typename Tuple, typename Func, std::size_t... I>
void ForEachTupleElement(const Tuple &tuple, Func func, std::index_sequence<I...>) {
  (func(std::get<I>(tuple)), ...);
}

static std::vector<ppc::util::PerfTestParam<InType, OutType>> BuildPerfTestParams() {
  const auto tuple_tasks = ppc::util::MakeAllPerfTasks<InType, SizovDGlobalSearchMPI, SizovDGlobalSearchSEQ>(
      PPC_SETTINGS_sizov_d_global_search);

  constexpr std::size_t kSize = std::tuple_size_v<decltype(tuple_tasks)>;

  std::vector<ppc::util::PerfTestParam<InType, OutType>> params;
  params.reserve(kSize);

  ForEachTupleElement(tuple_tasks, [&params](const auto &task_tuple) {
    const auto &task_getter = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTaskGetter)>(task_tuple);
    const auto &name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(task_tuple);
    const auto &mode = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(task_tuple);

    params.emplace_back(task_getter, name, mode);
  }, std::make_index_sequence<kSize>{});

  return params;
}

class SizovDGlobalSearchPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = MakePerfProblem();
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &out) override {
    if (!out.converged) {
      return false;
    }
    if (!std::isfinite(out.value)) {
      return false;
    }
    if (out.argmin < input_data_.left - 1e-6 || out.argmin > input_data_.right + 1e-6) {
      return false;
    }
    return true;
  }

 private:
  InType input_data_{};
};

class PerfTestInstance final : public SizovDGlobalSearchPerfTests {
 public:
  explicit PerfTestInstance(ppc::util::PerfTestParam<InType, OutType> param) : param_(std::move(param)) {}

  void TestBody() override {
    ExecuteTest(param_);
  }

 private:
  ppc::util::PerfTestParam<InType, OutType> param_;
};

struct PerfTestRegistrar {
  explicit PerfTestRegistrar(std::vector<ppc::util::PerfTestParam<InType, OutType>> tasks) : tasks_(std::move(tasks)) {
    test_names_.reserve(tasks_.size());

    std::size_t idx = 0;
    for (const auto &param : tasks_) {
      const auto &name_test = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(param);
      const auto &mode = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);

      const std::string test_name = ppc::performance::GetStringParamName(mode) + "_" + name_test;

      test_names_.push_back(test_name);
      const std::size_t current_idx = idx++;

      ::testing::RegisterTest(
          "GlobalSearchPerfTests", test_names_.back().c_str(), nullptr, nullptr, __FILE__, __LINE__,
          [this, current_idx]() -> ::testing::Test * { return new PerfTestInstance(tasks_[current_idx]); });
    }
  }

 private:
  std::vector<ppc::util::PerfTestParam<InType, OutType>> tasks_;
  std::vector<std::string> test_names_;
};

const bool kRegisteredPerfTests = []() {
  static const PerfTestRegistrar registrar(BuildPerfTestParams());
  (void)registrar;
  return true;
}();

}  // namespace sizov_d_global_search
