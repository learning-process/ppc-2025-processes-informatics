#include "gonozov_l_simple_iteration_method/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <tuple>
#include <cmath>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_simple_iteration_method {

GonozovLSimpleIterationMethodSEQ::GonozovLSimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  number_unknowns = static_cast<int>(std::get<0>(GetInput()));
  std::vector gOutp(number_unknowns, 0.0);
  GetOutput() = gOutp;
}

bool GonozovLSimpleIterationMethodSEQ::ValidationImpl() {
  // д.б. |a11| > |a12|+|a13|, |a22| > |a21|+|a23|, |a33| > |a31|+|a32|
  return (static_cast<int>(std::get<0>(GetInput())) > 0) && (static_cast<int>(std::get<1>(GetInput()).size()) > 0) && (static_cast<int>(std::get<2>(GetInput()).size()) > 0);
}

bool GonozovLSimpleIterationMethodSEQ::PreProcessingImpl() {
  return true;
}

bool GonozovLSimpleIterationMethodSEQ::RunImpl() {
    //number_unknowns = std::get<0>(GetInput());
    int max_number_iteration = 10000;
    
    // Получаем входные данные
    std::vector<double> matrix = std::get<1>(GetInput());
    std::vector<double> b = std::get<2>(GetInput());
    
    // Векторы для метода Якоби
    std::vector<double> previous_approximations(number_unknowns, 0.0);
    std::vector<double> current_approximations(number_unknowns, 0.0);
    
    // 1. Нулевое приближение: x_i^(0) = b_i / a_ii
    for (int i = 0; i < number_unknowns; i++) {
        previous_approximations[i] = b[i] / matrix[i * number_unknowns + i];
    }
    
    // 2. Основной цикл метода Якоби
    for (int iter = 0; iter < max_number_iteration; iter++) {
        // Для каждой переменной вычисляем новое приближение
        for (int i = 0; i < number_unknowns; i++) {
            double sum = 0.0;
            
            // Суммируем все недиагональные элементы: Σ a_ij * x_j^(k) для j ≠ i
            for (int j = 0; j < number_unknowns; j++) {
                if (j != i) {
                    sum += matrix[i * number_unknowns + j] * previous_approximations[j];
                }
            }
            
            // Новое приближение: x_i^(k+1) = (b_i - Σ) / a_ii
            current_approximations[i] = (b[i] - sum) / matrix[i * number_unknowns + i];
        }
        
        // 3. Проверка сходимости
        int converged = 0;
        for (int i = 0; i < number_unknowns; i++) {
            double diff = fabs(current_approximations[i] - previous_approximations[i]);
            double norm = fabs(current_approximations[i]);
            if (diff < 0.00001 * (norm + 1e-10)) {
                converged++;
            }
        }
        
        // Если все переменные сошлись
        if (converged == number_unknowns) {
            break;
        }
        
        // 4. Подготовка к следующей итерации
        previous_approximations = current_approximations;
    }
    
    // 5. Сохраняем результат
    for (int i = 0; i < number_unknowns; i++) {
        GetOutput()[i] = current_approximations[i];
    }
    
    return true;
  // std::vector<double> previous_approximations(number_unknowns, 0.0);

  // std::vector<double> matrix = std::get<1>(GetInput());
  // std::vector<double> b = std::get<2>(GetInput());
  // // построение нулевого приближения
  // for (int i = 0; i < number_unknowns; i++)
  // {
  //   previous_approximations[i] = (b[i] / matrix[i + number_unknowns * i]);
  // }

  // std::vector<double> current_approximations(number_unknowns, 0.0);

  // bool flag = true;
  // do
  // {
  //   for (int i = 0; i < number_unknowns; i++)
  //   {
  //     current_approximations[i] = (b[i] - matrix[(i+1) % number_unknowns + number_unknowns * i] 
  //     * previous_approximations[(i + 1) % number_unknowns]
  //     - matrix[(i+2) % number_unknowns + number_unknowns * i] 
  //     * previous_approximations[(i + 2) % number_unknowns]) / matrix[i + number_unknowns * i];
  //   }
    
  //   int count = 0;
  //   for (int i = 0; i < number_unknowns; i++)
  //   {
  //     if (abs(current_approximations[i] - previous_approximations[i]) / current_approximations[i] < 0.00001)
  //         count++;
  //   }
  //   if (count == number_unknowns)
  //     flag = false;
  //   previous_approximations = current_approximations;
  // } while (flag);

  // for (int i = 0; i < static_cast<int>(current_approximations.size()); i++)
  //   GetOutput()[i] = current_approximations[i];

  // return true;
}

bool GonozovLSimpleIterationMethodSEQ::PostProcessingImpl() {
  return GetOutput().size() > 0;
}

}  // namespace nesterov_a_test_task_processes_2
