# Быстрая сортировка с простым слиянием.

- Student: Маслова Ульяна Александровна, group 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 14

## 1. Introduction
Проблема: Сортировка больших массивов данных является ресурсоемкой задачей. Стандартные последовательные алгоритмы работают медленно на большом количестве данных.
Задача: Ускорить этот процесс с помощью параллельных вычислений на MPI.
Ожидаемый результат: Значительное сокращение времени выполнения по сравнению с последовательной версией.

## 2. Problem Statement
Дан массив чисел, нужно его отсортировать используя быструю сортировку с простым слиянием.
- InPut: std::vector<int>.
- OutPut: std::vector<int> (отсортированный).

## 3. Baseline Algorithm (Sequential)
Последовательная версия просто копирует входные данные и вызывает стандартную функцию std::sort.

## 4. Parallelization Scheme
Сначала главный процесс берет исходный большой массив и разделяет его на равные части. Эти кусочки раздаются всем доступным процессам. Далее каждый процесс сортирует полученную часть данных. Затем происходит сборка результатов. Процессы разбиваются на пары: один передает свои данные соседу, а сосед аккуратно соединяет два отсортированных списка в один общий, сохраняя порядок элементов. На следующем шаге оставшиеся процессы снова объединяются в пары. Это повторяется до тех пор, пока весь полностью отсортированный массив не соберется у главного процесса.

## 5. Implementation Details
- common: Определяет общие типы данных (InType, OutType).
- seq: Содержит простую последовательную реализацию алгоритма.
- mpi: Содержит параллельную MPI-реализацию алгоритма.
- tests: Включает два набора тестов: functional для проверки корректности и performance для замера скорости.

## 6. Experimental Setup
- Аппаратное обеспечение: AMD Ryzen 7 7840HS (8 ядер, 16 логических процессоров, базовая частота 3,80 ГГц)
- ОЗУ — 16 ГБ 
- Операционная система: Windows 11
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность проверялась на различных массивах (пустые массивы, одиночные элементы, массивы с повторениями, уже отсортированные и обратно отсортированные массивы).

### 7.2 Performance

Тест на данных, состоящих из 20 000 000 элементов:

| Mode | Count | Time, s   | Speedup | Efficiency |
|------|-------|-----------|---------|------------|
| seq  | 1     | 1.710     | 1.00    | N/A        |
| mpi  | 2     | 0.985     | 1.74    | 87.0%      |
| mpi  | 4     | 0.622     | 2.75    | 68.8%      |
| mpi  | 8     | 0.461     | 3.71    | 46.4%      |

## 8. Conclusions
Реализация параллельной сортировки продемонстрировала уверенное ускорение на больших объемах данных (20 млн элементов). С увеличением числа процессов время выполнения стабильно снижается. На 2 процессах достигнуто ускорение 1.74x. На 4 процессах — 2.75x. На 8 процессах — 3.71x. Это отличный результат.


## 9. References
1. Лекции, практики курса "Параллельное программирование"

## Appendix (Optional)
```cpp
bool MaslovaUFastSortSimpleMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    return true;
  }

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int part = total_size / size;
  int rem = total_size % size;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = part + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_vec(send_counts[rank]);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_vec.data(),
               send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  std::sort(local_vec.begin(), local_vec.end());

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      if (rank + step < size) {
        int recv_size = 0;
        MPI_Status status;
        MPI_Probe(rank + step, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &recv_size);
        std::vector<int> received(recv_size);
        MPI_Recv(received.data(), recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> merged(local_vec.size() + received.size());
        std::merge(local_vec.begin(), local_vec.end(), received.begin(), received.end(), merged.begin());
        local_vec = std::move(merged);
      }
    } else {
      int target = rank - step;
      MPI_Send(local_vec.data(), (int)local_vec.size(), MPI_INT, target, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }

  if (rank == 0) {
    GetOutput() = std::move(local_vec);
  }
  return true;
}
```