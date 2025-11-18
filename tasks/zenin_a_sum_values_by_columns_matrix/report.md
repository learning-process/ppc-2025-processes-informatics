# Сумма значений по столбцам матрицы
**Студент:** Зенин Антон Алексеевич.
**Группа:** 3823Б1ФИ1
**Технология:** SEQ | MPI
**Вариант:** 12 
---

## 1. Введение
Цель задачи состоит в том, чтобы вычислить сумму значений в каждом столбце матрицы, используя два подхода:

1. **Sequential (SEQ)** реализация
2. **Parallel MPI реализация** с использованием нескольких процессов

Целью работы является: 
- разработка и внедрение обеих версий,
- проверка корректности с помощью функциональных тестов,
- измерение и анализ производительности,
- оценка параллельной эффективности и выявление узких мест.

---

## 2. Постановка задачи

Дана матрица `A` размером `R × C`, представленная в виде вектора в порядке row-major. R - число строк, C - число столбцов. Нужно вычислить:

$$
S_j = \sum_{i=1}^{R} A_{i,j},\quad j = 1..C
$$

**Входные данные**

- `InType = tuple<size_t rows, size_t columns, vector<double> data>`  
- Размер данных = `rows * columns`

**Выходные данные** 

- `OutType = vector<double>` of size `columns`

Особенности: 

- Элементы имеют тип `double`
- Входной формат проверяется перед вычислением 

--- 

## 3. Описание алгоритма SEQ (Sequential)

Последовательный алгоритм выполняет простой вложенный цикл:

for each row:
for each column:
result[column] += A[row][column]

Временная сложность:

$$
O(R \cdot C)
$$

Использование памяти:

- Входная матрица: `R*C*sizeof(double)`
- Выходные данные: `C*sizeof(double)`

Для performance тестов размер матрицы был: **4000 x 4000 (16,000,000 элементов)**

---

## 4. Схема распараллеливания (MPI)

## Стратегия

Данные матрицы **передаются** всем MPI процессам.
Каждый процесс обрабатывает **непересекающееся подмножество столбцов**:

- `columns_per_proc = columns / world_size`
- Последний процесс получает остаток. 

Каждый процесс вычисляет частичные суммы для своих локальных столбцов: 

$$
S^{(rank)}_j = \sum_{i=1}^{R} A_{i,j}
$$

Затем происходит сбор результатов с помощью **MPI_Gatherv** в процессе с рангом 0, получая полный выходной вектор:

$$
S = \bigcup_{rank=0}^{P-1} S^{(rank)}
$$

В конце результат **передаётся** всем процессам.

## Схема взаимодействия

1. `MPI_Bcast` — передача данных о матрице процессам
2. `MPI_Bcast` — передача всей матрицы всем процессам
3. Локальные вычисления
4. `MPI_Gatherv` — сбор частичных результатов на процессе с рангом 0
5. `MPI_Bcast` — передача итогового результата

---

## 5. Особенности реализации

## Структура кода

common/include/common.hpp — InType, OutType, формат входных и выходных данных  
seq/src/ops_seq.cpp — реализация последовательного алгоритма (SEQ)  
mpi/src/ops_mpi.cpp — реализация паралелльного алгоритма (MPI)  
tests/functional/main.cpp — functional tests  
tests/performance/main.cpp — performance tests  

## Классы

- `ZeninASumValuesByColumnsMatrixSEQ : BaseTask`
- `ZeninASumValuesByColumnsMatrixMPI : BaseTask`

## Соображения памяти

- Вся матрица передаётся на все процессы -> потребление памяти: 128 МБ на процесс.
- Дополнительные временные буферы минимальны: каждый процесс хранит только  
локальный фрагмент выходного вектора (частичные суммы столбцов), что является очень небольшим объёмом по сравнению с размером матрицы.


---

## 6. Окружение

### Hardware
- CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz  
- Cores: 6  
- RAM: 32 GB  
- OS: Windows 10 Pro

### Toolchain
- Compiler: `C++20`   
- MPI: OpenMPI  
- Build type: Release

### Environment variables

PPC_NUM_THREADS = 1    
PPC_NUM_PROCS = 4 (для запуска с использованием MPI)

## Данные
- Функциональные тесты используют 9 входных файлов: `matrix1.txt`, `matrix2.txt`, `matrix3.txt`, `matrix4.txt`, `matrix5.txt`, `matrix6.txt`,   
`matrix7.txt`, `matrix8.txt`, `matrix9.txt`.   
- Тест на производительность использует файл `mat_perf.txt` размером **4000×4000** 

---

## Результаты и выводы

### 7.1 Корректность
Корректность была протестирована с помощью функциональных тестов:

- 9 входных матриц
- Обе реализации SEQ и MPI были проверены
- Все тесты были пройдены как в однопроцессорном, так и в многопроцессорном режимах
- Выходные данные сравниваются с ожидаемыми суммами с погрешностью `1e-12`

### 7.2 Производительность

#### Результаты performance-тестов

| Mode (normal run) | Time (s) | Speedup vs SEQ | Efficiency (4 proc) |
|-------------------|----------|----------------|---------------------|
| **SEQ pipeline**  | 0.011576 |      1.00      |          —          |
| **SEQ task_run**  | 0.011932 |      0.97      |          —          |
| **MPI pipeline**  | 0.142752 |      0.08      |         1.9%        |
| **MPI task_run**  | 0.146573 |      0.08      |         2.0%        |

---

| Mode (mpiexec -n 4) | Time (s) | Speedup vs SEQ | Efficiency (4 proc) |
|---------------------|----------|----------------|---------------------|
| **SEQ pipeline**    | 0.019669 |       1.00     |          —          |
| **SEQ task_run**    | 0.013334 |       1.47     |          —          |
| **MPI pipeline**    | 0.132601 |       0.15     |         3.7%        |
| **MPI task_run**    | 0.202401 |       0.10     |         2.4%        |

---

## Анализ результатов

1. **Релизация MPI выполняется медленнее, чем SEQ**, поскольку:
   - В реализации выполняется `MPI_Bcast` всей матрицы, поэтому каждый процесс хранит полный буфер матрицы.  
   Передача и распределение такого объёма данных создает значителньные накладные расходы.
   - Сбор результатов `MPI_Gatherv` сам по себе небольшой, но суммарные коммуникации (broadcast + gather) и синхронизация всё равно дают ощутимые накладные расходы.
   - Разложение по столбцам приводит к низкому отношению вычислений к коммуникации: на каждый процесс приходится сравнительно мало арифметики, тогда как коммуникационные затраты велики.
2. **Локальность памяти и порядок доступа**.
- В SEQ-реализации внешний цикл по строкам, внутренний — по столбцам: код читает элементы подряд в памяти `matrix[row*columns + col]`, т.е. использует    строчную (row-major) локальность, что эффективно использует кеш.
- В MPI-реализации после распределения стобцов каждый процесс для фиксированного `global_col` итерирует по `row`, обращаясь к элементам с шагом `columns`.  Такиой доступ хуже использует кеш и дополнительно снижает производительность.
- Накладные расходы на коммуникации и синхронизацию могут перекрывать выгоду от параллелизма.
---

## 8. Выводы

- Реализация параллельного алгоритма с использованием MPI для задачи "Сумма значений по столбцам матрицы" показала **низкую эффективность**.
- Последовательная реализация является высокоэффективной и превосходит MPI по производительности из-за отсутствия затрат на коммуникацию.
- Версия MPI демонстрирует **очень низкое ускорение (0,08–0,15)** и низкую эффективность из-за:
  - большого количества операций передачи,
  - низкого соотношения вычислений к передаче данных,
  - высоких накладных расходов. 

Несмотря на это, реализация MPI является **корректной, стабильной и полностью функциональной** и проходит все тесты.

---

## 9. Источники
- cppreference.com - https://en.cppreference.com
- Документация по OpenMPI - https://www.open-mpi.org/doc
- Лекции по параллельному программированию ННГУ им. Лобачевского
- Практические занятия по параллельному программированию ННГУ им. Лобачевского

---

## 10. Приложение. Код MPI реализации
```cpp

#pragma once

#include "task/include/task.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByColumnsMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninASumValuesByColumnsMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::tuple<size_t, size_t> CalculateProcessColumns(int rank, int world_size, size_t columns);
  void CalculateLocalSums(const std::vector<double>& matrix_data, size_t columns,
                         size_t total_rows, size_t start_column, 
                         size_t cols_this_process, std::vector<double>& local_sums);
  void PrepareGathervParameters(int world_size, size_t base_cols_per_process,
                               size_t remain, std::vector<int>& recv_counts,
                               std::vector<int>& displacements);

};

}  // namespace zenin_a_sum_values_by_columns_matrix

std::tuple<size_t, size_t> ZeninASumValuesByColumnsMatrixMPI::CalculateProcessColumns(int rank, int world_size, size_t columns) {
  size_t base_cols_per_process = columns / world_size;
  size_t remain = columns % world_size;

  size_t start_column = rank * base_cols_per_process;
  size_t cols_this_process = base_cols_per_process;

  if (rank == world_size - 1) {
    cols_this_process += remain;
  }
  return {start_column, cols_this_process};
}

void ZeninASumValuesByColumnsMatrixMPI::CalculateLocalSums(const std::vector<double> &matrix_data, 
  size_t columns, size_t total_rows, size_t start_column, size_t cols_this_process, std::vector<double> &local_sums) {
  for (size_t local_column = 0; local_column < cols_this_process; ++local_column) {
    size_t global_col = start_column + local_column;
    for (size_t row = 0; row < total_rows; ++row) {
      local_sums[local_column] += matrix_data[(row * columns) + global_col];
    }
  }
}

void ZeninASumValuesByColumnsMatrixMPI::PrepareGathervParameters(int world_size, size_t base_cols_per_process, 
  size_t remain, std::vector<int> &recv_counts, std::vector<int> &displacements) {
  for (int i = 0; i < world_size; ++i) {
    recv_counts[i] = static_cast<int>(base_cols_per_process);
    if (i == world_size - 1) {
      recv_counts[i] += static_cast<int>(remain);
    }
    if (i > 0) {
      displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }
  }
}

bool ZeninASumValuesByColumnsMatrixMPI::RunImpl() {
  auto &input = GetInput();
  int world_size = 0;
  int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t columns = 0;
  std::vector<double> matrix_data;
  size_t total_rows = 0;

  if (rank == 0) {
    columns = std::get<1>(input);
    matrix_data = std::get<2>(input);
    total_rows = std::get<0>(input);
    if (matrix_data.size() % columns != 0) {
      return false;
    }
  }

  MPI_Bcast(&columns, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (columns == 0) {
    return false;
  }

  if (rank != 0) {
    matrix_data.resize(total_rows * columns);
  }
  MPI_Bcast(matrix_data.data(), static_cast<int>(matrix_data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto [start_column, cols_this_process] = CalculateProcessColumns(rank, world_size, columns);
  std::vector<double> local_sums(cols_this_process, 0.0);
  CalculateLocalSums(matrix_data, columns, total_rows, start_column, cols_this_process, local_sums);

  std::vector<double> global_sums;
  if (rank == 0) {
    global_sums.resize(columns, 0.0);
  }

  size_t base_cols_per_process = columns / world_size;
  size_t remain = columns % world_size;

  std::vector<int> recv_counts(world_size, 0);
  std::vector<int> displacements(world_size, 0);

  if (rank == 0) {
    PrepareGathervParameters(world_size, base_cols_per_process, remain, recv_counts, displacements);
  }

  MPI_Gatherv(local_sums.data(), static_cast<int>(local_sums.size()), MPI_DOUBLE, global_sums.data(),
              recv_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  
  if (rank != 0) {
    global_sums.resize(columns);
  }

  MPI_Bcast(global_sums.data(), static_cast<int>(columns), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = global_sums;

  return true;
}