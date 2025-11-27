#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "titaev_m_avg_el_vector/common/include/common.hpp"
#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"
#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace titaev_m_avg_el_vector {

using SizeParam = std::tuple<int>;

using DescriptorType = ppc::util::PerfTestParam<InType, OutType>;
using TestParamType = std::tuple<DescriptorType, int>;

class TitaevMAvgElVectorPerfTest : public ::testing::TestWithParam<TestParamType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<TestParamType> &info) {
    const DescriptorType &desc = std::get<0>(info.param);
    const int size = std::get<1>(info.param);
    const std::string base_name = std::get<1>(desc);
    return base_name + "_Size_" + std::to_string(size);
  }

 protected:
  void SetUp() override {
    const int size = std::get<1>(this->GetParam());
    input_data_.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-1000, 1000);
    for (int i = 0; i < size; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  InType GetTestInputData() {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &) {
    return true;
  }

  void ExecutePerf(const DescriptorType &perf_test_param) {
    auto task_getter = std::get<0>(perf_test_param);
    auto test_name = std::get<1>(perf_test_param);
    auto mode = std::get<2>(perf_test_param);

    ASSERT_FALSE(test_name.find("unknown") != std::string::npos);
    if (test_name.find("disabled") != std::string::npos) {
      GTEST_SKIP();
    }

    const auto test_env_scope = ppc::util::test::MakePerTestEnvForCurrentGTest(test_name);

    auto task = task_getter(GetTestInputData());
    ppc::performance::Perf perf(task);
    ppc::performance::PerfAttr perf_attr;

    if (task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kMPI ||
        task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kALL) {
      const double t0 = ppc::util::GetTimeMPI();
      perf_attr.current_timer = [t0] { return ppc::util::GetTimeMPI() - t0; };
    } else if (task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kOMP) {
      const double t0 = omp_get_wtime();
      perf_attr.current_timer = [t0] { return omp_get_wtime() - t0; };
    } else if (task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kSEQ ||
               task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kSTL ||
               task->GetDynamicTypeOfTask() == ppc::task::TypeOfTask::kTBB) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      perf_attr.current_timer = [t0] {
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        return static_cast<double>(ns) * 1e-9;
      };
    } else {
      std::stringstream err_msg;
      err_msg << '\n' << "The task type is not supported for performance testing.\n";
      throw std::runtime_error(err_msg.str().c_str());
    }

    if (mode == ppc::performance::PerfResults::TypeOfRunning::kPipeline) {
      perf.PipelineRun(perf_attr);
    } else if (mode == ppc::performance::PerfResults::TypeOfRunning::kTaskRun) {
      perf.TaskRun(perf_attr);
    } else {
      std::stringstream err_msg;
      err_msg << '\n' << "The type of performance check for the task was not selected.\n";
      throw std::runtime_error(err_msg.str().c_str());
    }

    if (ppc::util::GetMPIRank() == 0) {
      perf.PrintPerfStatistic(test_name);
    }

    OutType output_data = task->GetOutput();
    ASSERT_TRUE(CheckTestOutputData(output_data));
  }

 private:
  InType input_data_;
};

namespace {

const std::array<int, 4> kSizes = {10000, 100000, 1000000, 10000000};

static std::vector<TestParamType> BuildPerfParams() {
  std::vector<TestParamType> res;

  const auto mpi_tasks =
      ppc::util::MakePerfTaskTuples<TitaevMAvgElVectorMPI, InType>(PPC_SETTINGS_titaev_m_avg_el_vector);
  const auto seq_tasks =
      ppc::util::MakePerfTaskTuples<TitaevMAvgElVectorSEQ, InType>(PPC_SETTINGS_titaev_m_avg_el_vector);

  const DescriptorType mpi_desc0 = std::get<0>(mpi_tasks);
  const DescriptorType mpi_desc1 = std::get<1>(mpi_tasks);

  const DescriptorType seq_desc0 = std::get<0>(seq_tasks);
  const DescriptorType seq_desc1 = std::get<1>(seq_tasks);

  for (int s : kSizes) {
    res.emplace_back(std::make_tuple(mpi_desc0, s));
    res.emplace_back(std::make_tuple(mpi_desc1, s));
    res.emplace_back(std::make_tuple(seq_desc0, s));
    res.emplace_back(std::make_tuple(seq_desc1, s));
  }

  return res;
}

static const std::vector<TestParamType> kPerfParams = BuildPerfParams();

TEST_P(TitaevMAvgElVectorPerfTest, ParallelVectorAverage) {
  const auto param = this->GetParam();
  const DescriptorType &desc = std::get<0>(param);
  ExecutePerf(desc);
}

INSTANTIATE_TEST_SUITE_P(AvgVectorPerfTests, TitaevMAvgElVectorPerfTest, ::testing::ValuesIn(kPerfParams),
                         TitaevMAvgElVectorPerfTest::PrintTestParam);

}  // namespace

}  // namespace titaev_m_avg_el_vector
