# Сумма значений по столбцам матрицы

- Студент: Санников Иван Михайлович, Группа: 3823Б1ФИ2
- Технология: SEQ, MPI
- Вариант: 12
## 1. Введение
Вычисление сумм по столбцам матрицы является одной из основных в математике и часто используется в прикладных задачах, как машинное обучение(ML) и статистика. Часто приходится работать с большим объемом данных, что делает задачу ресурсоемкой при использовании последовательных алгоритмов. Цель данной лабораторной работы - написание алгоритма на основе технологии MPI для распределение нагрузки между несколькоми процессами. 

## 2. Постановка задачи
Входные данные: std::vector<std::vector<int>> - вектор векторов типа данных int представляющий из себя матрицу.

Для матрицы размера A*B, строица std::vector<int> размера B, где для элемента i хранится сумма столбца i из входной матрицы.

## 3. Базовый алгоритм(seq)
Последовательный алгорритм:
- Проходим по всем строкам
- В каждой i строке проходим по всем столбцам
- Складываем элемент i строки j столбца c элементом j вектора суммы.

```cpp
bool SannikovIColumnSumSEQ::RunImpl() {
  const auto &input_matrix = GetInput();
  if (input_matrix.empty()) {
    return false;
  }

  for (int i = 0; i < (int)input_matrix.size(); i++) {
    for (int j = 0; j < (int)input_matrix[i].size(); j++) {
      GetOutput()[j] += input_matrix[i][j];
    }
  }
  return !GetOutput().empty();
}
```

## 4. Описание параллельного алгоритма 

1. Обрабатываем входные данные: Превращаем std::vector<std::vector<int>> в последовательный std::vector<int>, чтобы была возможность его разделить Scatterv.
2. Вычисляем сколько данных получит процесс: перемножаем количество столбцов на строки, делим на количество процессов и прибавляем 1, если номер вычисления меньше остатка (i<rem?1:0). Получив массив элементов сдвига вычисляем сам сдвиг для определенного rank. Сдвиг равен элементу массива сдвига по номеру rank и вычисляем его остаток от деления на количество столбцов (id_elem[rank] % columns).
3. Рассылаем данные MPI_Scatterv.
4. Вычисляем локальные суммы: Создается локальный вектор сум. При помощи вычисленных сдвигов заполняется значениями сум столбцов входной матрицы.
5. Собираются все данные при помощи MPI_Allreduce.

## 5. Experimental Setup

- Hardware/OS: Intel i9 13900KF, 24 ядра, RAM: 32Gb, OS: Windows 11
- Toolchain: Cmake 3.28.3, g++ (Ubuntu 14.2.0 x86_64), Docker-контейнер, Режим сборки: Release.
- Data: Для тестов на производительность использовалась матрица размером 10000 на 10000 заполненнаная по алгоритму: элемент i строки j столбца равен i * 14 + j * 21.

## 6. Результаты

### 6.1 Корректность 

Корректность работы алгоритма проверена при помощи технологии Google Test. Для проверки использовались входные данные в виде матриц разных размеров, пустые матрицы и матрицы с отрицательными числами. 

### 6.2 Производительность

Входные данные: Матрица 10000 на 10000.

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.020   | 1.00    | N/A        |
| mpi         | 2     | 0.218   | 0.09    | 4.59%      |
| mpi         | 4     | 0.160   | 0.12    | 3.10%      |
| mpi         | 8     | 0.150   | 0.12    | 1.60%      |

## 7. Выводы

Из результатов тестов на производительность можно сделать вывод, что данная задача не подходит для реализации на технологии MPI. Затраты системы на обработку вызовов Scatterv и Allreduce прривышают суммарные затраты на обработку данных в последовательном алгоритме. 

## 8. Литература
1. Open MPI: Documentation - https://www.open-mpi.org/doc/
2. Александер Сысоев. Курс лекций по параллельному программированию. Лекция № 3 - https://cloud.unn.ru/s/o5y2jGbxb7XpBJa
3. Parallel Programming 2025-2026 - https://disk.yandex.ru/d/NvHFyhOJCQU65w

## 9. Приложение

```cpp
bool SannikovIColumnSumMPI::RunImpl() {
  const auto &input_matrix = GetInput();

  int rank = 0;
  int size = 1;
  int rows = 0;
  int columns = (GetOutput().size());

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    if (input_matrix.empty() || (int)input_matrix.front().size() != columns) {
      return false;
    }
    rows = (input_matrix.size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows <= 0 || columns <= 0) {
    return false;
  }
  std::vector<int> sendbuf;
  if (rank == 0) {
    sendbuf.resize((rows * columns));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        sendbuf[i * columns + j] = input_matrix[i][j];
      }
    }
  }
  std::vector<int> elem_for_proc(size);
  std::vector<int> id_elem(size);
  int displacement = 0;
  for (int i = 0; i < size; i++) {
    elem_for_proc[i] = rows * columns / size + (i < rows * columns % size ? 1 : 0);
    id_elem[i] = displacement;
    displacement += elem_for_proc[i];
  }
  int mpi_displacement = id_elem[rank] % columns;
  std::vector<int> buf(elem_for_proc[rank], 0);
  MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, elem_for_proc.data(), id_elem.data(), MPI_INT, buf.data(),
               elem_for_proc[rank], MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum(columns, 0);
  for (int i = 0; i < elem_for_proc[rank]; i++) {
    int new_col = (i + mpi_displacement) % columns;
    sum[new_col] += buf[i];
  }
  MPI_Allreduce(sum.data(), GetOutput().data(), columns, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return !GetOutput().empty();
}


```