#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_topology_star/common/include/common.hpp"
#include "zenin_a_topology_star/mpi/include/ops_mpi.hpp"
#include "zenin_a_topology_star/seq/include/ops_seq.hpp"

namespace zenin_a_topology_star {

class ZeninATopologyStarFunctTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t rows = std::get<0>(params);
    size_t cols = std::get<1>(params);
    std::vector<double> mat(rows * cols);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        mat[(i * cols) + j] = (static_cast<double>(i) + static_cast<double>(j)) * 0.5;
      }
    }
    input_data_ = std::make_tuple(rows, cols, mat);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t rows = std::get<0>(input_data_);
    const std::vector<double> &mat = std::get<2>(input_data_);
    size_t cols = std::get<1>(input_data_);
    std::vector<double> correct(cols, 0.0);

    // std::vector<double> expected_sums(columns, 0.0);

    for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
        correct[j] += mat[(i * cols) + j];
      }
    }

    return (correct == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ZeninATopologyStarFunctTests, Test) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(3, 3),     std::make_tuple(2, 5),   std::make_tuple(10, 70),     std::make_tuple(1, 1),
    std::make_tuple(1, 100),   std::make_tuple(100, 1), std::make_tuple(1000, 1000), std::make_tuple(10, 2),
    std::make_tuple(5, 3),     std::make_tuple(4, 5),   std::make_tuple(4, 3),       std::make_tuple(10000, 3),
    std::make_tuple(3, 10000), std::make_tuple(500, 1), std::make_tuple(1, 500)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZeninATopologyStarMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zenin_a_topology_star),
                                           ppc::util::AddFuncTask<ZeninATopologyStarSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zenin_a_topology_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZeninATopologyStarFunctTests::PrintFuncTestName<ZeninATopologyStarFunctTests>;

INSTANTIATE_TEST_SUITE_P(ZeninATopologyStar, ZeninATopologyStarFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zenin_a_topology_star
