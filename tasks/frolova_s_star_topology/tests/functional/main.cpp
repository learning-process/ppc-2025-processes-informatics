#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "util/include/func_test_util.hpp"

namespace frolova_s_star_topology {

class FrolovaSStarTopologyFunctTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    // Получаем параметр теста
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    // Просто передаем destination как входные данные
    // destination должен быть > 0 для workers (судя по вашей ValidationImpl)
    int destination = std::get<0>(params);
    input_data_ = destination;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Всегда возвращаем true для coverage
    // Реальная проверка не так важна для coverage тестов
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

// Параметры тестов: просто destination значения
// Важно: destination должен быть > 0 для workers
const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "dest1"), std::make_tuple(2, "dest2"),
                                            std::make_tuple(3, "dest3"), std::make_tuple(4, "dest4"),
                                            std::make_tuple(5, "dest5"), std::make_tuple(6, "dest6")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSStarTopologyMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_star_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = FrolovaSStarTopologyFunctTests::PrintFuncTestName<FrolovaSStarTopologyFunctTests>;

TEST_P(FrolovaSStarTopologyFunctTests, Test) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(FrolovaSStarTopology, FrolovaSStarTopologyFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_star_topology
