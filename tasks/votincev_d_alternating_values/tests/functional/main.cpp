#include <gtest/gtest.h>
#include <stb/stb_image.h>

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

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatingValuesFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // чтобы тесты по названию не пересекались
  static std::string PrintTestParam(const TestType &test_param) {
    std::string strV = "Vector: ";
    for (int i = 0; i < test_param.size(); i++) {
      strV += std::to_string(test_param[i]);
      strV += " ";
    }
    strV += "\n";

    return strV;
  }

 protected:
  // считываем/генерируем данные
  void SetUp() override {
    int sz = swaps_count + 1;
    std::vector<double> v(sz);
    int swapper = 1;
    for (int i = 0; i < sz; i++) {
      v.push_back(i * swapper);  // 0 -1 2 -3 4 -5...
      swapper *= -1;
    }
    input_data_ = v;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == swaps_count;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType swaps_count = 400000;  // добавил
};

namespace {

TEST_P(VotincevDAlternatingValuesFuncTests, CountSwapsFromGenerator) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::vector<double>{0, -1, 2, -3}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VotincevDAlternatingValuesMPI, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values),
                                           ppc::util::AddFuncTask<VotincevDAlternatingValuesSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VotincevDAlternatingValuesFuncTests::PrintFuncTestName<VotincevDAlternatingValuesFuncTests>;

// запускаем параметрически тест
INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VotincevDAlternatingValuesFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace votincev_d_alternating_values
