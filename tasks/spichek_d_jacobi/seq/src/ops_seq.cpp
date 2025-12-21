#include "spichek_d_jacobi/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"

namespace spichek_d_jacobi {

SpichekDJacobiSEQ::SpichekDJacobiSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Vector{};
}

bool SpichekDJacobiSEQ::ValidationImpl() {
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    return true;
  }
  if (A[0].size() != n || b.size() != n) {
    return false;
  }
  // Простая проверка диагонального преобладания для валидации
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(A[i][i]) < 1e-12) {
      return false;
    }
  }
  return true;
}

bool SpichekDJacobiSEQ::PreProcessingImpl() {
  return true;
}

bool SpichekDJacobiSEQ::RunImpl() {
  // // ВРЕМЕННЫЙ отладочный вывод
  // std::cout << "[SEQ DEBUG] RunImpl() начал выполнение" << std::endl;

  const auto &[A, b, eps_input, max_iter_input] = GetInput();
  size_t n = A.size();

  // std::cout << "[SEQ DEBUG] Размер матрицы n = " << n << std::endl;
  // std::cout << "[SEQ DEBUG] epsilon = " << eps_input << std::endl;
  // std::cout << "[SEQ DEBUG] max_iter = " << max_iter_input << std::endl;

  if (n == 0) {
    // std::cout << "[SEQ DEBUG] n == 0, возвращаем пустой вектор" << std::endl;
    GetOutput() = Vector{};
    return true;
  }

  Vector x_k(n, 0.0);
  Vector x_k_plus_1(n, 0.0);

  int iter = 0;
  double max_diff;

  // std::cout << "[SEQ DEBUG] Начинаем итерации Якоби" << std::endl;

  do {
    ++iter;
    max_diff = 0.0;

    // if (iter % 10 == 0) {  // Выводим каждые 10 итераций
    //   std::cout << "[SEQ DEBUG] Итерация " << iter << std::endl;
    // }
    if (iter > max_iter_input) {
      //      std::cout << "[SEQ DEBUG] ПРЕВЫШЕН max_iter! iter = " << iter
      //              << ", max_iter_input = " << max_iter_input << std::endl;
      break;
    }

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      const auto &row = A[i];

      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += row[j] * x_k[j];
        }
      }

      x_k_plus_1[i] = (b[i] - sum) / row[i];
    }

    for (size_t i = 0; i < n; ++i) {
      max_diff = std::max(max_diff, std::abs(x_k_plus_1[i] - x_k[i]));
    }

    // if (iter % 10 == 0) {  // Выводим каждые 10 итераций
    //   std::cout << "[SEQ DEBUG] max_diff = " << max_diff
    //             << ", eps_input = " << eps_input << std::endl;
    // }

    x_k = x_k_plus_1;

  } while (max_diff > eps_input && iter < max_iter_input);

  // std::cout << "[SEQ DEBUG] Завершили итерации. Всего итераций: " << iter << std::endl;
  // std::cout << "[SEQ DEBUG] Финальный max_diff = " << max_diff << std::endl;
  // std::cout << "[SEQ DEBUG] Возвращаем результат размером " << x_k.size() << std::endl;

  // // Проверим несколько первых элементов результата
  // std::cout << "[SEQ DEBUG] Первые 5 элементов результата: ";
  // for (int i = 0; i < std::min(5, (int)n); ++i) {
  //   std::cout << x_k[i] << " ";
  // }
  // std::cout << std::endl;

  GetOutput() = x_k;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
