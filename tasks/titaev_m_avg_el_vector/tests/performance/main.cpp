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

using PerfTestParam = std::tuple<int>;

class TitaevMAvgElVectorPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<typename TitaevMAvgElVectorPerfTest::ParamType> &info) {
    const PerfTestParam &test_param = std::get<0>(info.param);
    return "Size_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    const PerfTestParam &params = std::get<0>(this->GetParam());
    int size = std::get<0>(params);

    input_data_.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-1000, 1000);

    for (int i = 0; i < size; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &) override {
    return true;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TitaevMAvgElVectorPerfTest, ParallelVectorAverage) {
  ExecuteTest(this->GetParam());
}

const std::array<PerfTestParam, 4> kTestParam_perf = {std::make_tuple(10000), std::make_tuple(100000),
                                                      std::make_tuple(1000000), std::make_tuple(10000000)};

const auto kTestTasksList_perf =
    std::tuple_cat(ppc::util::AddPerfTask<TitaevMAvgElVectorMPI>(kTestParam_perf, PPC_SETTINGS_titaev_m_avg_el_vector),
                   ppc::util::AddPerfTask<TitaevMAvgElVectorSEQ>(kTestParam_perf, PPC_SETTINGS_titaev_m_avg_el_vector));

const auto kGtestValues_perf = ppc::util::ExpandToValues<PerfTestParam>(kTestTasksList_perf);

const auto kPerfTestName = TitaevMAvgElVectorPerfTest::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(AvgVectorPerfTests, TitaevMAvgElVectorPerfTest, kGtestValues_perf, kPerfTestName);

}  // namespace

}  // namespace titaev_m_avg_el_vector
