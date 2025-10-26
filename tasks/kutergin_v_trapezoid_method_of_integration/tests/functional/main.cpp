#include <gtest/gtest.h>

#include <vector>
#include <tuple>

#include "seq/kutergin_v_trapezoid_method_of_integration/include/trapezoid_integration_sequential.hpp"
#include "util/include/func_test_util.hpp"


namespace kutergin_v_trapezoid_seq
{


using TestType = std::tuple<InputData, double, std::string>; // Тип для тестовых параметров {входные данные, ожидаемый результат, имя_теста}


class KuterginVRunFuncTestsSEQ : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> // наследник ppc::util::BaseRunFuncTests<InType, OutType, TestType> 
{
public:

    static std::string PrintTestParam(const TestType& test_param) // вывод имен тестов
    {
        return std::get<2>(test_param);
    }

protected:

    // подготовка входных данных перед каждым тестом
    void SetUp() override
    {
        TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()); // получение параметров для текущего теста
        input_data_ = std::get<0>(params); // извлечение входных данных
        expected_output_ = std::get<1>(params); // извлечение ожидаемого результата
    }

    // проверка результата после выполнения задачи
    bool CheckTestOutputData(OutType& output_data) final 
    {
        return std::abs(output_data - expected_output_) < 1e-6;
    }

    // возврат подготовленных входных данных для задачи
    InType GetTestInputData() final
    {
        return input_data_;
    }

private:

    InType input_data_{};
    OutType expected_output_{};
};

namespace // анонимное пространство имен
{
    
TEST_P(KuterginVRunFuncTestsSEQ, TrapezoidTest) // параметризованный тест
{
    ExecuteTest(GetParam());
}

// массив с наборами тестовых данных
const std::array<TestType, 2> kTestCases = {
    // тест 1
    std::make_tuple(InputData{0.0, 3.0, 100}, 9.0, "f(x)=x^2_[0,3]_n=100"), 
    // тест 2
    std::make_tuple(InputData{-1.0, 1.0, 200}, 0.666666, "f(x)=x^2_[-1,1]_n=200")
};

// используем фреймворк для подготовки задач к запуску
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TrapezoidIntegrationSequential, InType>(kTestCases, ""));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KuterginVRunFuncTestsSEQ::PrintFuncTestName<KuterginVRunFuncTestsSEQ>;

// "регистрация" набора тестов и параметров в GTest
INSTANTIATE_TEST_SUITE_P(TrapezopidIntegrationSEQ, KuterginVRunFuncTestsSEQ, kGtestValues, kTestName); 
}

}


