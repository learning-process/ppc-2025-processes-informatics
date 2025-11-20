# Отчёт по лабораторной работе  
## «Параллельное нахождение максимальных значений по столбцам матрицы с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Косолапов Виталий Андреевич 3823Б1ФИ1
**Вариант:** 16
---

## Введение

Задача нахождения максимальных значений по столбцам матрицы является фундаментальной операцией в вычислительной математике и анализе данных. Эта операция широко используется в различных приложениях, включая обработку сигналов, машинное обучение, научные вычисления и бизнес-аналитику. В контексте обработки больших данных, когда матрицы могут содержать миллионы элементов, последовательный алгоритм становится недостаточно эффективным. Одним из ключевых методов ускорения вычислений является распараллеливание. В данной работе рассматривается реализация алгоритма нахождения максимальных значений по столбцам матрицы, как в последовательной, так и в параллельной формах, с использованием технологии MPI (Message Passing Interface).  

Цель работы — продемонстрировать принципы параллельного программирования, а также сравнить эффективность и корректность работы MPI-версии с последовательной реализацией.

---

## Постановка задачи

Необходимо реализовать нахождение **максимального значения по всем столбцам матрицы**.  

Дана матрица A размером M × N найти вектор максимумов V размером N, где каждый элемент V[i] представляет максимальное значение в i-м столбце матрицы A.

$$
V[i] = max{A[0][i], A[1][i], ..., A[M-1][i]} для всех i ∈ [0, N-1]
$$

### Требуется:
1. Реализовать **последовательную версию** вычисления.  
2. Реализовать **параллельную версию** с использованием **MPI**.  
3. Провести сравнение времени выполнения и подтвердить корректность вычислений.  

---

## Описание алгоритма

Алгоритм поиска максимального значения по столбцам матрицы можно описать следующим образом:

1. Проверка корректности входных данных: все строки матрицы должны быть одной длины.  
2. Инициализация результата пустым вектором длины N.  
3. Для всех элементов \( i \) от 0 до \( N-1 \): 
   - Установить значение `temp_max` = A[0][i];
   - Для всех элементов \( j \) от 0 до \( M-1 \):
     - Если A[j][i] > `temp_max`:
       - Установить значение `temp_max` = A[j][i];  
   - Установить i-й элемент результирующего вектора равным `temp_max`;
4. Вернуть полученное значение как результат поиска максимума.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `processes_count`.  
2. **Разделение данных:**  
   - Столбцы равномерно распределяются между процессами.  
   - Если количество столбцов не делится нацело, лишние элементы распределяются по первым процессам.  
3. **Вычисление частичных результатов:**  
   Каждый процесс вычисляет максимальное значение своей части столбцов.  
4. **Сбор результатов:**  
   Процесс с `rank` равным 0 собирает данные с остальных процессов, используя `MPI_Recv`, вычисляет итоговый результат, а затем отправляетв его другим процессам, используя `MPI_Send`. Процессы чей `rank` отличен от 0 отправляют локальный результат на процессор с `rank` равным 0, используя `MPI_Send`, затем они собирают результат с этого процесса, используя `MPI_Recv`. 
5. **Вывод результата и завершение работы MPI.**

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++17** с использованием **библиотеки MPI**.  
Класс `KosolapovVMaxValuesInColMatrixMPI` реализует интерфейс параллельного поиска максимальных значений.  

Основные этапы:
- Инициализация и валидация входных данных;  
- Определение диапазона элементов, обрабатываемых каждым процессом;  
- Локальное вычисление максимумов;  
- Объединение частичных результатов между всеми процессами при помощи `MPI_Recv` и `MPI_Send`;  
- Возврат итогового значения.

### Ключевые функции:
- `MPI_Comm_rank`, `MPI_Comm_size` — определяют номер процесса и общее количество процессов;  
- `MPI_Send` — отправляет сообщение(данные) другому процессу;
- `MPI_Recv` — принимает сообщение(данные) от другого процесса.

### Валидация данных
Не допускаются матрицы с разными длинами строк.

---

## Результаты экспериментов

### Условия экспериментов
- Размеры матриц: \(\(10^5\)*\(10^5\)\) и \(\(10^4\)*\(10^4\)\) элементов.  
- Среда выполнения: Windows, MPI (4 процесса).  
- Измерение времени проводилось встроенными средствами тестового фреймворка GoogleTest.  

---

### Результаты при размере матрицы \(\(10^4\)*\(10^4\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.00587     | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.00597     | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.00168     | 3.49      | 87.25%        |
| MPI task_run    | 4                | 0.00159     | 3.75      | 93.75%        |
| MPI pipeline    | 5                | 0.00134     | 4.38      | 87.6%         |
| MPI task_run    | 5                | 0.00129     | 4.63      | 92.6%         |

---

### Результаты при размере матрицы \(\(10^5\)*\(10^5\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.890       | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.915       | 1.0       | N/A           |
| MPI pipeline    | 2                | 0.416       | 2.14      | 107%          |
| MPI task_run    | 2                | 0.419       | 2.18      | 109%          |
| MPI pipeline    | 4                | 0.238       | 3.73      | 93.25%        |
| MPI task_run    | 4                | 0.233       | 3.93      | 98.25%        |
| MPI pipeline    | 6                | 0.183       | 4.86      | 81%           |
| MPI task_run    | 6                | 0.191       | 4.79      | 79.83%        |

---

## Подтверждение корректности
Результаты последовательной и параллельной версий совпадают для всех протестированных вариаций матриц, включая граничный случаи(матрица 1*1, матрица с одной строчкой, матрица с одним столбцом). Ошибок не наблюдалось, при тестировании возвращался ожидаемый результат. Таким образом, реализация корректна.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется и даёт ускорение при увеличении размера данных.  
3. При малых объёмах данных накладные расходы MPI превышают выигрыш от распараллеливания.  
4. Алгоритм корректен и демонстрирует ожидаемое поведение при всех размерах входных данных.  
5. Эффективность увеличивается, при увелечение входных данных, но снижается при увелечение количества процессов, что связано с увелечение наклодных расходов.

---

## Заключение
В ходе лабораторной работы был реализован и протестирован алгоритм поиска максимального элемента по столбцам матрицы с использованием технологии MPI. Проведён сравнительный анализ с последовательной версией, подтверждена корректность вычислений, проведены измерения ускорения и эффективности. Полученные результаты демонстрируют эффективность использования параллельных вычислений для ресурсоёмких задач.

---

## Список литературы

1. Параллельные вычисления. MPI Tutorial. — https://mpitutorial.com  
2. Документация по OpenMPI — https://www.open-mpi.org/doc/ 
3. Лекции Сысоева Александра Владимировича
4. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича

---

## Приложение. Исходный код

### Последовательная версия

```cpp

#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixSEQ::KosolapovVMaxValuesInColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  for (size_t i = 0; i < matrix.size() - 1; i++) {
    if (matrix[i].size() != matrix[i + 1].size()) {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  for (size_t i = 0; i < matrix[0].size(); i++) {
    int temp_max = matrix[0][i];
    for (const auto &row : matrix) {
      temp_max = std::max(row[i], temp_max);
    }
    GetOutput()[i] = temp_max;
  }
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kosolapov_v_max_values_in_col_matrix
```
---
### MPI-версия
```cpp
#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixMPI::KosolapovVMaxValuesInColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  for (size_t i = 0; i < matrix.size() - 1; i++) {
    if (matrix[i].size() != matrix[i + 1].size()) {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int columns = static_cast<int>(matrix[0].size());

  const int columns_per_proc = columns / processes_count;
  const int remainder = columns % processes_count;

  const int start = (rank * columns_per_proc) + std::min(rank, remainder);

  auto local_maxs = CalculateLocalMax(matrix, rank, processes_count, columns);
  std::vector<int> global_maxs;
  if (rank == 0) {
    global_maxs.resize(columns);
    for (size_t i = 0; i < local_maxs.size(); i++) {
      global_maxs[start + i] = local_maxs[i];
    }

    for (int proc = 1; proc < processes_count; proc++) {
      const int proc_start = (proc * columns_per_proc) + std::min(proc, remainder);
      const int proc_end = proc_start + columns_per_proc + (proc < remainder ? 1 : 0);
      const int proc_columns_count = proc_end - proc_start;
      std::vector<int> proc_maxs(proc_columns_count);
      MPI_Recv(proc_maxs.data(), proc_columns_count, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int i = 0; i < proc_columns_count; i++) {
        global_maxs[proc_start + i] = proc_maxs[i];
      }
    }
    for (int proc = 1; proc < processes_count; proc++) {
      MPI_Send(global_maxs.data(), columns, MPI_INT, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    global_maxs.resize(columns);
    MPI_Recv(global_maxs.data(), columns, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  GetOutput() = global_maxs;
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::PostProcessingImpl() {
  return true;
}

std::vector<int> KosolapovVMaxValuesInColMatrixMPI::CalculateLocalMax(const std::vector<std::vector<int>> &matrix,
                                                                      int rank, int processes_count, int columns) {
  const int columns_per_proc = columns / processes_count;
  const int remainder = columns % processes_count;

  const int start = (rank * columns_per_proc) + std::min(rank, remainder);
  const int end = start + columns_per_proc + (rank < remainder ? 1 : 0);
  std::vector<int> local_maxs(end - start);
  for (int i = start; i < end; i++) {
    int temp_max = matrix[0][i];
    for (const auto &row : matrix) {
      temp_max = std::max(row[i], temp_max);
    }
    local_maxs[i - start] = temp_max;
  }
  return local_maxs;
}
}  // namespace kosolapov_v_max_values_in_col_matrix
```