#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "dolov_v_torus_topology/mpi/include/ops_mpi.hpp"
#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int world_size = 1;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_type = std::get<0>(params);

    std::vector<int> message = {42, 13, 7};
    // Инициализируем входные данные с учетом поля total_procs
    input_data_.sender_rank = 0;
    input_data_.total_procs = world_size;
    input_data_.message = message;

    int r = static_cast<int>(std::sqrt(world_size));
    while (world_size % r != 0) {
      r--;
    }
    int c = world_size / r;

    expected_.received_message = message;

    switch (test_type) {
      case 0:  // Self
        input_data_.receiver_rank = 0;
        expected_.route = {0};
        break;
      case 1:  // East (Step or Wrap)
        input_data_.receiver_rank = (c > 1) ? 1 : 0;
        if (world_size > 1 && c > 1) {
          expected_.route = {0, 1};
        } else {
          expected_.route = {0};
        }
        break;
      case 2:  // South (Step or Wrap)
        input_data_.receiver_rank = (r > 1) ? c : 0;
        if (world_size > 1 && r > 1) {
          expected_.route = {0, c};
        } else {
          expected_.route = {0};
        }
        break;
      case 3:  // West Boundary (Wrap-around)
        input_data_.receiver_rank = (c > 1) ? (c - 1) : 0;
        if (c > 1) {
          expected_.route = {0, c - 1};
        } else {
          expected_.route = {0};
        }
        break;
      case 4:  // North Boundary (Wrap-around)
        input_data_.receiver_rank = (r > 1) ? (r - 1) * c : 0;
        if (r > 1) {
          expected_.route = {0, (r - 1) * c};
        } else {
          expected_.route = {0};
        }
        break;
      case 5:  // Far Destination (Diagonal/Opposite)
      default:
        input_data_.receiver_rank = world_size - 1;
        // Для сложного пути полагаемся на валидацию начала и конца в CheckTestOutputData,
        // так как расчет точного маршрута DOR здесь продублирует RunImpl.
        expected_.route.clear();
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1. Проверка сообщения
    if (output_data.received_message != expected_.received_message) {
      return false;
    }

    // 2. Проверка маршрута
    if (output_data.route.empty()) {
      return false;
    }

    // Если ожидаемый маршрут задан явно (для простых случаев)
    if (!expected_.route.empty()) {
      return output_data.route == expected_.route;
    }

    // Для сложных маршрутов (FarDestination) проверяем корректность начала и конца
    return output_data.route.front() == input_data_.sender_rank &&
           output_data.route.back() == input_data_.receiver_rank;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(DolovVTorusTopologyFuncTests, TorusRouting) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "Self"),          std::make_tuple(1, "NeighborEast"),
                                            std::make_tuple(2, "NeighborSouth"), std::make_tuple(3, "BoundaryWest"),
                                            std::make_tuple(4, "BoundaryNorth"), std::make_tuple(5, "FarDestination")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<DolovVTorusTopologyMPI, InType>(kTestParam, PPC_SETTINGS_dolov_v_torus_topology),
    ppc::util::AddFuncTask<DolovVTorusTopologySEQ, InType>(kTestParam, PPC_SETTINGS_dolov_v_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(TorusFunctional, DolovVTorusTopologyFuncTests, kGtestValues,
                         DolovVTorusTopologyFuncTests::PrintFuncTestName<DolovVTorusTopologyFuncTests>);

}  // namespace
}  // namespace dolov_v_torus_topology
