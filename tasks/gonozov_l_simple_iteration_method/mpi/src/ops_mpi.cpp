#include "gonozov_l_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_simple_iteration_method {

GonozovLSimpleIterationMethodMPI::GonozovLSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool GonozovLSimpleIterationMethodMPI::ValidationImpl() {
  // д.б. |a11| > |a12|+|a13|, |a22| > |a21|+|a23|, |a33| > |a31|+|a32|
  number_unknowns = std::get<0>(GetInput());
  return (static_cast<int>(std::get<0>(GetInput())) > 0) && (static_cast<int>(std::get<1>(GetInput()).size()) > 0) &&
         (static_cast<int>(std::get<2>(GetInput()).size()) > 0);
}

bool GonozovLSimpleIterationMethodMPI::PreProcessingImpl() {
  return true;
}

bool GonozovLSimpleIterationMethodMPI::RunImpl() {
  // int proc_num = 0;
  // int proc_rank = 0;
  // MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  // MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // if (number_unknowns < proc_num) {
  //   std::vector<double> current_approximations(number_unknowns, 0.0);
  //   if (proc_rank == 0) {
  //     std::vector<double> previous_approximations(number_unknowns, 0.0);
  //     std::vector<double> matrix = std::get<1>(GetInput());
  //     std::vector<double> b = std::get<2>(GetInput());
  //     // построение нулевого приближения
  //     for (int i = 0; i < number_unknowns; i++)
  //     {
  //       previous_approximations[i] = (b[i] / matrix[i + number_unknowns * i]);
  //     }

  //     bool flag = true;
  //     do
  //     {
  //       for (int i = 0; i < number_unknowns; i++)
  //       {
  //         current_approximations[i] = (b[i] - matrix[(i+1) % number_unknowns + number_unknowns * i]
  //         * previous_approximations[(i + 1) % number_unknowns]
  //         - matrix[(i+2) % number_unknowns + number_unknowns * i]
  //         * previous_approximations[(i + 2) % number_unknowns]) / matrix[i + number_unknowns * i];
  //       }

  //       int count = 0;
  //       for (int i = 0; i < number_unknowns; i++)
  //       {
  //         if (abs(current_approximations[i] - previous_approximations[i]) / current_approximations[i] < 0.00001)
  //             count++;
  //       }
  //       if (count == number_unknowns)
  //         flag = false;
  //       previous_approximations = current_approximations;
  //     } while (flag);

  //     MPI_Bcast(current_approximations.data(), current_approximations.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   }
  //   for (int i = 0; i < static_cast<int>(current_approximations.size()); i++)
  //     GetOutput()[i] = current_approximations[i];
  //   return true;
  // }

  // std::vector<double> matrix;
  // std::vector<double> b;

  //  if (proc_rank == 0) {
  //   matrix = std::get<1>(GetInput());
  //   b = std::get<2>(GetInput());
  // }

  // int number_processed = number_unknowns / proc_num;
  // int remainder = number_unknowns % proc_num;
  // int local_size = number_processed + (proc_rank < remainder ? 1 : 0);

  // std::vector<int> sendcounts_for_matrix(proc_num);
  // std::vector<int> sendcounts_for_b(proc_num);
  // std::vector<int> displs_for_matrix(proc_num);
  // std::vector<int> displs_for_b(proc_num);
  // std::vector<double> strings_of_matrix(local_size * number_unknowns);
  // std::vector<double> block_of_b(local_size);

  // if (proc_rank == 0) {
  //   int offset_for_matrix = 0;
  //   int offset_for_b = 0;
  //   for (int i = 0; i < proc_num; i++) {
  //     sendcounts_for_matrix[i] = number_unknowns * (number_processed + (i < remainder ? 1 : 0));
  //     displs_for_matrix[i] = offset_for_matrix;
  //     offset_for_matrix += sendcounts_for_matrix[i];

  //     sendcounts_for_b[i] = number_processed + (i < remainder ? 1 : 0);
  //     displs_for_b[i] = offset_for_b;
  //     offset_for_b += sendcounts_for_b[i];
  //   }
  // }

  // MPI_Scatterv((proc_rank == 0) ? matrix.data() : nullptr, sendcounts_for_matrix.data(), displs_for_matrix.data(),
  // MPI_DOUBLE,
  //             strings_of_matrix.data(), local_size * number_unknowns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // MPI_Scatterv((proc_rank == 0) ? b.data() : nullptr, sendcounts_for_b.data(), displs_for_b.data(), MPI_DOUBLE,
  //             block_of_b.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // std::vector<double> previous_approximations(number_unknowns, 0.0);
  // std::vector<double> local_previous_approximations(local_size, 0.0);
  // std::vector<double> current_approximations(number_unknowns, 0.0);
  // std::vector<double> local_current_approximations(local_size, 0.0);

  // //вычисление добавочного значения для определения диагональных элементов
  // int64_t additional_value_diag;
  // if (proc_rank < remainder)
  //   additional_value_diag = proc_rank * local_size;
  // else
  //   additional_value_diag = remainder + local_size * proc_rank;

  // // на каждом процессе происходит построение local_size нулевых приближений
  // for (int i = 0; i < local_size; i++)
  // {
  //   local_previous_approximations[i] = (block_of_b[i] / strings_of_matrix[i + additional_value_diag + number_unknowns
  //   * i]);
  // }

  // MPI_Bcast(sendcounts_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Bcast(displs_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);

  // MPI_Allgatherv(local_previous_approximations.data(), local_size, MPI_DOUBLE,
  //    previous_approximations.data(), sendcounts_for_b.data(), displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  // //int64_t flag_current;//признак завершения подсчёта для текущего процесса
  // do
  // {
  //   for (int i = 0; i < local_size; i++)
  //   {
  //     local_current_approximations[i] = (block_of_b[i]
  //     - strings_of_matrix[(i + additional_value_diag + 1) % number_unknowns + number_unknowns * i]
  //     * previous_approximations[(i + 1) % number_unknowns]
  //     - strings_of_matrix[(i + additional_value_diag + 2) % number_unknowns + number_unknowns * i]
  //     * previous_approximations[(i + 2) % number_unknowns]) / strings_of_matrix[i + additional_value_diag +
  //     number_unknowns * i];
  //   }

  //   MPI_Allgatherv(local_current_approximations.data(), local_size, MPI_DOUBLE,
  //       current_approximations.data(), sendcounts_for_b.data(), displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  //   int local_count = 0;
  //   for (int i = 0; i < local_size; i++)
  //   {
  //       if (abs(current_approximations[i] - previous_approximations[i]) < 0.00001 * abs(current_approximations[i]))
  //           local_count++;
  //   }

  //   int global_count = 0;
  //   MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  //   if (global_count == number_unknowns)
  //     break;

  //   previous_approximations = current_approximations;
  //   local_previous_approximations = local_current_approximations;
  // } while (true);

  // MPI_Bcast(current_approximations.data(), current_approximations.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // for (int i = 0; i < static_cast<int>(current_approximations.size()); i++)
  //   GetOutput()[i] = current_approximations[i];

  // return true;
  int max_number_iteration = 10000;
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // // Если неизвестных меньше процессов
  // if (number_unknowns < proc_num) {
  //     if (proc_rank == 0) {
  //         // Последовательный метод Якоби
  //         std::vector<double> previous(number_unknowns, 0.0);
  //         std::vector<double> matrix = std::get<1>(GetInput());
  //         std::vector<double> b = std::get<2>(GetInput());

  //         for (int i = 0; i < number_unknowns; i++) {
  //             previous[i] = b[i] / matrix[i * number_unknowns + i];
  //         }

  //         std::vector<double> current(number_unknowns, 0.0);

  //         for (int iter = 0; iter < max_number_iteration; iter++) {
  //             for (int i = 0; i < number_unknowns; i++) {
  //                 double sum = 0.0;
  //                 for (int j = 0; j < number_unknowns; j++) {
  //                     if (j != i) {
  //                         sum += matrix[i * number_unknowns + j] * previous[j];
  //                     }
  //                 }
  //                 current[i] = (b[i] - sum) / matrix[i * number_unknowns + i];
  //             }

  //             int converged = 0;
  //             for (int i = 0; i < number_unknowns; i++) {
  //                 double diff = abs(current[i] - previous[i]);
  //                 double norm = abs(current[i]);
  //                 if (diff < 0.00001 * (norm + 1e-10)) {
  //                     converged++;
  //                 }
  //             }

  //             if (converged == number_unknowns) {
  //                 std::cout << "Sequential: Converged in " << iter + 1 << " iterations" << std::endl;
  //                 break;
  //             }
  //             previous = current;
  //         }

  //         GetOutput() = current;
  //     }

  //     return true;
  // }

  // ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ С Scatterv
  std::vector<double> matrix;
  std::vector<double> b;

  if (proc_rank == 0) {
    matrix = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
    number_unknowns = std::get<0>(GetInput());
  }

  // Рассылаем размерность
  MPI_Bcast(&number_unknowns, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  // Распределение строк по процессам
  int number_processed = number_unknowns / proc_num;
  int remainder = number_unknowns % proc_num;
  int local_size = number_processed + (proc_rank < remainder ? 1 : 0);

  // Определяем, какие строки обрабатывает этот процесс
  int my_first_row = 0;
  for (int p = 0; p < proc_rank; p++) {
    my_first_row += number_processed + (p < remainder ? 1 : 0);
  }

  // Подготовка данных для Scatterv
  std::vector<int> sendcounts_for_matrix(proc_num, 0);
  std::vector<int> sendcounts_for_b(proc_num, 0);
  std::vector<int> displs_for_matrix(proc_num, 0);
  std::vector<int> displs_for_b(proc_num, 0);

  if (proc_rank == 0) {
    int offset_matrix = 0;
    int offset_b = 0;
    for (int p = 0; p < proc_num; p++) {
      int p_size = number_processed + (p < remainder ? 1 : 0);

      // Каждый процесс получает p_size строк матрицы
      // Каждая строка содержит number_unknowns элементов
      sendcounts_for_matrix[p] = p_size * number_unknowns;

      // Каждый процесс получает p_size элементов вектора b
      sendcounts_for_b[p] = p_size;

      displs_for_matrix[p] = offset_matrix;
      displs_for_b[p] = offset_b;

      offset_matrix += sendcounts_for_matrix[p];
      offset_b += sendcounts_for_b[p];
    }
  }

  // Распространяем информацию о распределении
  MPI_Bcast(sendcounts_for_matrix.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs_for_matrix.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendcounts_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);

  // Локальные буферы
  std::vector<double> local_matrix(local_size * number_unknowns);
  std::vector<double> local_b(local_size);

  // Разбрасываем данные
  MPI_Scatterv((proc_rank == 0) ? matrix.data() : nullptr, sendcounts_for_matrix.data(), displs_for_matrix.data(),
               MPI_DOUBLE, local_matrix.data(), local_size * number_unknowns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv((proc_rank == 0) ? b.data() : nullptr, sendcounts_for_b.data(), displs_for_b.data(), MPI_DOUBLE,
               local_b.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Векторы для метода Якоби
  std::vector<double> previous_approximations(number_unknowns, 0.0);
  std::vector<double> current_approximations(number_unknowns, 0.0);
  std::vector<double> local_previous(local_size, 0.0);
  std::vector<double> local_current(local_size, 0.0);

  // Нулевое приближение для моих строк
  for (int i = 0; i < local_size; i++) {
    int global_row = my_first_row + i;
    // Диагональный элемент находится в local_matrix[i * number_unknowns + global_row]
    // Но аккуратнее: каждая локальная строка начинается с элемента для столбца 0
    local_previous[i] = local_b[i] / local_matrix[i * number_unknowns + global_row];
  }

  // Собираем все начальные приближения
  MPI_Allgatherv(local_previous.data(), local_size, MPI_DOUBLE, previous_approximations.data(), sendcounts_for_b.data(),
                 displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  // Основной цикл метода Якоби (ОБЩИЙ СЛУЧАЙ)
  for (int iter = 0; iter < max_number_iteration; iter++) {
    // Каждый процесс вычисляет новые значения для своих строк
    for (int i = 0; i < local_size; i++) {
      int global_row = my_first_row + i;
      double sum = 0.0;

      // Важно: local_matrix[i * number_unknowns + j] содержит элемент
      // i-й локальной строки (которая соответствует глобальной строке global_row)
      // в столбце j

      // Суммируем все недиагональные элементы
      for (int j = 0; j < number_unknowns; j++) {
        if (j != global_row) {
          // Доступ к элементу в i-й локальной строке, j-м столбце
          sum += local_matrix[i * number_unknowns + j] * previous_approximations[j];
        }
      }

      // Диагональный элемент: local_matrix[i * number_unknowns + global_row]
      local_current[i] = (local_b[i] - sum) / local_matrix[i * number_unknowns + global_row];
    }

    // Собираем все новые приближения
    MPI_Allgatherv(local_current.data(), local_size, MPI_DOUBLE, current_approximations.data(), sendcounts_for_b.data(),
                   displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Проверка сходимости
    int local_converged = 0;
    for (int i = 0; i < local_size; i++) {
      int global_row = my_first_row + i;
      double diff = abs(current_approximations[global_row] - previous_approximations[global_row]);
      double norm = abs(current_approximations[global_row]);
      if (diff < 0.00001 * (norm + 1e-10)) {
        local_converged++;
      }
    }

    int global_converged = 0;
    MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_converged == number_unknowns) {
      break;
    }

    // Подготовка к следующей итерации
    previous_approximations = current_approximations;
    local_previous = local_current;
  }

  GetOutput() = current_approximations;

  return true;
}

bool GonozovLSimpleIterationMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_simple_iteration_method
