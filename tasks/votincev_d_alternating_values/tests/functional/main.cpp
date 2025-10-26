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

class VotincevDAlternatigValuesRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // пока что Test Type == std::tuple<int, std::string>
  // test_param надо обязательно юзать!
  // изменил на std::string
  static std::string PrintTestParam(const TestType &test_param) {
    // std::string strV;
    // for (int i = 0; i < test_param.size(); i++) {
    //   strV += std::to_string(test_param[i]);
    //   strV += "_";
    // }
    // strV += "\n";
    return test_param;
  }

 protected:
  // считываем/генерируем данные
  void SetUp() override {
    int sz = expectedRes + 1;  // 5
    std::vector<double> v;
    int swapper = 1;
    for (int i = 0; i < sz; i++) {
      v.push_back(i * swapper);  // 0 -1 2 -3 4 ...
      swapper *= -1;
    }
    input_data_ = v;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 0-й процесс генерирует результат
    // остальные - выполняют работу
    // если будут процессы 1,2 ... сравнивать результат с корректным
    // то всегда будет неверно
    // поэтому можно или сделать как Дмитрий Сизов
    // (только 0й процесс сравнивает на корректность, остальные возвращают true)
    // или можно сделать как я:
    // 0й процесс в RunImpl распределяет выходные данные между остальными
    // остальные делают GetOutput() = выходной_результат
    // мой способ - хуже в плане производительности
    // но лучше в плане "честности"
    // (можно ведь написать огромный нерабочий код и выставить как рабочий)
    // (в моем случае если код не рабочий,то он не рабочий, у него нет выхода)

    // std::cout << "From CheckTestOutputData in FuncTests:_" <<  output_data << "\n";
    return output_data == expectedRes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expectedRes = 4;  // добавил
};

namespace {

TEST_P(VotincevDAlternatigValuesRunFuncTestsProcesses, CountSwapsFromGenerator) {
  ExecuteTest(GetParam());
}

// тестов всего k, но на самом деле 2k, на MPI и на SEQ,
// так что не удивляемся, когда видим их в 2 раза больше
const std::array<TestType, 1> kTestParam = {"myDefaultTest"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VotincevDAlternatingValuesMPI, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values),
                                           ppc::util::AddFuncTask<VotincevDAlternatingValuesSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VotincevDAlternatigValuesRunFuncTestsProcesses::PrintFuncTestName<VotincevDAlternatigValuesRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(CountSwapsFromGeneratorr, VotincevDAlternatigValuesRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace votincev_d_alternating_values
