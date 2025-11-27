#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "titaev_m_avg_el_vector/common/include/common.hpp"
#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"
#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace titaev_m_avg_el_vector {

class TitaevMAvgElVectorFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, FuncTestParam> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(testparam)) + "" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const FuncTestParam &params = std::get<2>(this->GetParam());
    int size = std::get<0>(params);

    input_data_.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-100, 100);

    for (int i = 0; i < size; ++i) {
      input_data_[i] = dist(gen);
    }

    if (size > 0) {
      int64_t sum = std::accumulate(input_data_.begin(), input_data_.end(), static_cast<int64_t>(0));
      reference_output_ = static_cast<double>(sum) / size;
    } else {
      reference_output_ = 0.0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(reference_output_ - output_data) < 1e-5;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType reference_output_ = 0.0;
};

namespace {

TEST_P(TitaevMAvgElVectorFuncTest, RandomVectorAverage) {
  ExecuteTest(this->GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(10, "Tiny"), std::make_tuple(100, "Small"),
                                            std::make_tuple(1000, "Medium"), std::make_tuple(5000, "Large")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TitaevMAvgElVectorMPI, InType>(kTestParam, PPC_SETTINGS_titaev_m_avg_el_vector),
    ppc::util::AddFuncTask<TitaevMAvgElVectorSEQ, InType>(kTestParam, PPC_SETTINGS_titaev_m_avg_el_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = TitaevMAvgElVectorFuncTest::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(AvgVectorFuncTests, TitaevMAvgElVectorFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace titaev_m_avg_el_vector
