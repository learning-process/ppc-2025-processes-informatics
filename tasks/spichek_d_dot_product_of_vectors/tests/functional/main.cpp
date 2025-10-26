#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
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
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);  // Используем первый параметр как размер векторов
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для скалярного произведения векторов [1,2,...,n] • [1,2,...,n]
    // результат должен быть равен сумме квадратов: 1² + 2² + ... + n²
    InType n = input_data_;
    InType expected_result = 0;
    for (InType i = 1; i <= n; i++) {
        expected_result += i * i;
    }
    
    return (output_data == expected_result);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SpichekDDotProductOfVectorsRunFuncTestsProcesses, DotProductTest) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: (размер_вектора, описание)
const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, "vector_size_3"), 
    std::make_tuple(5, "vector_size_5"), 
    std::make_tuple(7, "vector_size_7")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SpichekDDotProductOfVectorsMPI, InType>(kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors),
                   ppc::util::AddFuncTask<SpichekDDotProductOfVectorsSEQ, InType>(kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SpichekDDotProductOfVectorsRunFuncTestsProcesses::PrintFuncTestName<SpichekDDotProductOfVectorsRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(DotProductTests, SpichekDDotProductOfVectorsRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace spichek_d_dot_product_of_vectors
