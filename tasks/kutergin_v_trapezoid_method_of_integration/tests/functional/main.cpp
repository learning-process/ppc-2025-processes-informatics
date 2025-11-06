#include <gtest/gtest.h>
#include <mpi.h>

#include <tuple>
#include <vector>

#include "../../mpi/include/trapezoid_integration_mpi.hpp"
#include "../../seq/include/trapezoid_integration_sequential.hpp"
#include "util/include/func_test_util.hpp"

namespace kutergin_v_trapezoid_seq {

class KuterginVRunFuncTestsSEQ
    : public ppc::util::BaseRunFuncTests<InType, OutType,
                                         TestType>  // наследник ppc::util::BaseRunFuncTests<InType, OutType, TestType>
{
 public:
  static std::string PrintTestParam(const TestType &test_param)  // вывод имен тестов
  {
    return std::get<2>(test_param);
  }

 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(
        GetParam());                         // получение параметров для текущего теста
    input_data_ = std::get<0>(params);       // извлечение входных данных
    expected_output_ = std::get<1>(params);  // извлечение ожидаемого результата

    if (ppc::util::IsUnderMpirun()) {
      int process_count;
      MPI_Comm_size(MPI_COMM_WORLD, &process_count);

      if (process_count > 0) {
        // Берем базовое n из теста и "округляем" его вверх до ближайшего числа, кратному числу процессов
        int base_n = input_data_.n;
        if (base_n % process_count != 0) {
          input_data_.n = base_n + (process_count - (base_n % process_count));
        }
      }
    }
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0)  // проверку выполняет только нулевой процесс
    {
      std::cout << "-> RANK 0 EXPECTED: " << expected_output_ << std::endl;
      std::cout << "-> RANK 0 ACTUAL:   " << output_data << std::endl;
      return std::abs(output_data - expected_output_) < 1e-6;
    }
    return true;  // у всех остальных процессов все в порядке
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_{};
};

namespace  // анонимное пространство имен
{

TEST_P(KuterginVRunFuncTestsSEQ, TrapezoidTest)  // параметризованный тест
{
  ExecuteTest(GetParam());
}

// массив с наборами тестовых данных
const std::array<TestType, 2> kTestCases = {
    // тест 1
    std::make_tuple(InputData{0.0, 3.0, 10000}, 9.0, "f_x_squared_0_to_3_n_10000"),
    // тест 2
    std::make_tuple(InputData{-1.0, 1.0, 20000}, 0.666666, "f_x_squared_neg1_to_1_n_20000")};

// используем фреймворк для подготовки задач к запуску
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TrapezoidIntegrationSequential, InType>(
                       kTestCases, PPC_SETTINGS_kutergin_v_trapezoid_method_of_integration),
                   ppc::util::AddFuncTask<kutergin_v_trapezoid_mpi::TrapezoidIntegrationMPI, InType>(
                       kTestCases, PPC_SETTINGS_kutergin_v_trapezoid_method_of_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KuterginVRunFuncTestsSEQ::PrintFuncTestName<KuterginVRunFuncTestsSEQ>;

// "регистрация" набора тестов и параметров в GTest
INSTANTIATE_TEST_SUITE_P(TrapezoidIntegrationSEQ, KuterginVRunFuncTestsSEQ, kGtestValues, kTestName);
}  // namespace

}  // namespace kutergin_v_trapezoid_seq
