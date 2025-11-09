#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_dot_product_of_vectors {

class SpichekDDotProductOfVectorsRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[vectors, description] = test_param;
    return description;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);  // Получаем пару векторов
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[vector1, vector2] = input_data_;

    // Вычисляем ожидаемый результат
    int expected_result = 0;
    for (size_t i = 0; i < vector1.size(); ++i) {
      expected_result += vector1[i] * vector2[i];
    }

    return (output_data == expected_result);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SpichekDDotProductOfVectorsRunFuncTestsProcesses, DotProductTest) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: ((вектор1, вектор2), описание)
const std::array<TestType, 3> kTestParam = {
    std::make_tuple(std::make_pair(std::vector<int>{1, 2, 3}, std::vector<int>{4, 5, 6}), "vectors_3"),
    std::make_tuple(std::make_pair(std::vector<int>{1, 0, -1, 2}, std::vector<int>{2, 3, 4, -1}),
                    "vectors_4_with_negatives"),
    std::make_tuple(std::make_pair(std::vector<int>{5, 5, 5}, std::vector<int>{2, 2, 2}), "constant_vectors")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SpichekDDotProductOfVectorsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors),
                                           ppc::util::AddFuncTask<SpichekDDotProductOfVectorsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SpichekDDotProductOfVectorsRunFuncTestsProcesses::PrintFuncTestName<
    SpichekDDotProductOfVectorsRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(DotProductTests, SpichekDDotProductOfVectorsRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace spichek_d_dot_product_of_vectors
