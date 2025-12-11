#include <gtest/gtest.h>
#include <mpi.h>
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

    const size_t msg_size = std::get<0>(params);
    const size_t pattern = std::get<1>(params);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int center = 0;
    int src = 0;
    int dst = 0;
    if (world_size == 1) {
      src = dst = 0;
    } else {
      switch (pattern % 3) {
        case 0:
          src = center;
          dst = world_size - 1;
          break;
        case 1:
          src = world_size - 1;
          dst = center;
          break;
        default:
          if (world_size >= 3) {
            src = 1;
            dst = world_size - 1;
          } else {
            src = center;
            dst = world_size - 1;
          }
          break;
      }
    }

    std::vector<double> data(msg_size);
    for (size_t i = 0; i < msg_size; ++i) {
      data[i] = static_cast<double>(i + pattern);
    }
    input_data_ = std::make_tuple(static_cast<size_t>(src), static_cast<size_t>(dst), std::move(data));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const auto &in = input_data_;
    const int dst = static_cast<int>(std::get<1>(in));
    const auto &data = std::get<2>(in);
    if (world_rank == dst) {
      if (output_data.size() != data.size()) {
        return false;
      }
      for (size_t i = 0; i < data.size(); ++i) {
        if (output_data[i] != data[i]) {
          return false;
        }
      }
    } else {
      if (!output_data.empty()) {
        return false;
      }
    }
    return true;
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

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZeninATopologyStarMPI, InType>(kTestParam, PPC_SETTINGS_zenin_a_topology_star),
    ppc::util::AddFuncTask<ZeninATopologyStarSEQ, InType>(kTestParam, PPC_SETTINGS_zenin_a_topology_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZeninATopologyStarFunctTests::PrintFuncTestName<ZeninATopologyStarFunctTests>;

INSTANTIATE_TEST_SUITE_P(ZeninATopologyStar, ZeninATopologyStarFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zenin_a_topology_star
