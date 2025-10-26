#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "util/include/util.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsSEQ::SpichekDDotProductOfVectorsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SpichekDDotProductOfVectorsSEQ::PreProcessingImpl() {
  // Инициализация или подготовка данных перед выполнением
  // Для скалярного произведения векторов можно инициализировать векторы здесь
  GetOutput() = 0;  // Сбрасываем выходное значение
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::RunImpl() {
  //std::cout << "SEQ RunImpl START: input=" << GetInput() << std::endl;
  
  if (GetInput() == 0) {
    //std::cout << "SEQ RunImpl: input is 0, returning false" << std::endl;
    return false;
  }

  InType n = GetInput();
  //std::cout << "SEQ RunImpl: n=" << n << std::endl;
  
  // Вычисляем скалярное произведение напрямую без создания векторов
  InType dot_product = 0;
  for (InType i = 1; i <= n; ++i) {
    dot_product += i * i;  // [1,2,...,n] • [1,2,...,n] = 1² + 2² + ... + n²
    
    // Отладочный вывод каждые 100000 итераций
    //if (i % 100000 == 0) {
    // std::cout << "SEQ RunImpl: i=" << i << ", dot_product=" << dot_product << std::endl;
    //}
  }

  // Сохраняем результат
  GetOutput() = dot_product;
  //std::cout << "SEQ RunImpl END: output=" << GetOutput() << std::endl;

  return GetOutput() > 0;
}

bool SpichekDDotProductOfVectorsSEQ::PostProcessingImpl() {
  // Постобработка результата (если нужна)
  // В данном случае просто проверяем валидность результата
  return GetOutput() > 0;
}

}  // namespace spichek_d_dot_product_of_vectors
