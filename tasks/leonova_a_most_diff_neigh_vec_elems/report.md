# Нахождение наиболее отличающихся по значению соседних элементов вектора

- Студент: Леонова Анна Сергеевна, группа 3823Б1ФИ1
- Технология: SEQ | MPI
- Вариант: 8

## 1. Введение
В данной работе рассматривается задача поиска пары соседних элементов вектора с максимальной разностью значений. Эта задача имеет практическое применение в обработке сигналов, анализе временных рядов и вычислительной математике. Цель работы - разработка и сравнение производительности последовательной и параллельной MPI-реализаций алгоритма.

## 2. Постановка задачи
Для заданного вектора целых чисел v = [v₁, v₂, ..., vₙ] необходимо найти пару соседних элементов (vᵢ, vᵢ₊₁) с максимальным значением модуля разности |vᵢ - vᵢ₊₁| (i = 1, 2, ..., n - 1).

Формат входных данных: std::vector<int> - вектор с целыми значениями
Формат выходных данных: std::tuple<int, int> - пара элементов с максимальной разницей

Ограничения и особенности реализации:

- Вектор должен содержать хотя бы один элемент;
- При размере вектора n = 1 возвращается пара (v₁, v₁);
- При нескольких парах с одинаковой максимальной разницей возвращается первая найденная такая пара.

## 3. Базовый алгоритм (Последовательный)
Алгоритм последовательно проходит по всем парам соседних элементов вектора v и отслеживает максимальную по модулю разность:

max_diff = -1
for i от 0 до n-2:
    curr_diff = |v[i] - v[i+1]|
    если curr_diff > max_diff:
        max_diff = curr_diff
        best_pair = (v[i], v[i+1])

Результатом выполнения алгоритма будет пара элементов с максимальной разницей - best_pair.
Вычислительная сложность алгоритма - O(n).

## 4. Схема распараллеливания
Вектор делится на перекрывающиеся блоки между процессами, чтобы каждый процесс мог проверить все соседние пары в своей части вектора без необходимости дополнительного обмена граничными элементами. Если процессов слишком много - они учавствуют в алгоритме, но рассылают фиктивные данные, не влияющие на результат работы. Каждый процесс, который должен учавствовать в вычислениях, кроме может быть последнего, получает k = chunk_size + 1 элементов для проверки всех пар в своей части, где chunk_size = n / N. (N - количество процессов) Последний процесс получает ровно столько элементов, сколько осталось до конца вектора.

Процесс 0 распределяет блоки данных через MPI_Send, остальные получают через MPI_Recv:
- Процесс 0: [0...k] элементы;
- Процесс 1: [k...2k] элементы;
...
- Процесс N-1: [(N-1)*k...N] элементы.

Так, например, имеем три процесса и исходный вектор v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
- Процесс 0: получает элементы [0, 1, 2, 3] → проверяет пары: (0,1), (1,2), (2,3);
- Процесс 1: получает элементы [3, 4, 5, 6] → проверяет пары: (3,4), (4,5), (5,6); 
- Процесс 2: получает элементы [6, 7, 8, 9] → проверяет пары: (6,7), (7,8), (8,9).
В результате проверяются все пары соседних элементов.

Каждый процесс находит локальную пару с локальной максимальной разницей. Результаты собираются на процессе 0 через MPI_Gather. Процесс 0 находит глобальный максимум и результат рассылается всем процессам через MPI_Bcast.

## 5. Детали реализации
Структура кода:
- ppc-2025-processes-informatics/tasks/leonova_a_most_diff_neigh_vec_elems/seq/src/ops_seq.cpp - последовательная реализация;
- ppc-2025-processes-informatics/tasks/leonova_a_most_diff_neigh_vec_elems/mpi/src/ops_mpi.cpp - MPI реализация.

Тесты: 
- ppc-2025-processes-informatics/tasks/leonova_a_most_diff_neigh_vec_elems/tests/functional/main.cpp - функциональные;
- ppc-2025-processes-informatics/tasks/leonova_a_most_diff_neigh_vec_elems/tests/performance/main.cpp - на производительность.

## 6. Experimental Setup

### 6.1 Оборудование и ПО
- CPU: Intel Core i5-1135G7 (4 ядра, 8 потоков);
- RAM: 8ГБ;
- ОС: Windows 10 Pro 22H2;
- Тип сборки: Release.

### 6.2 Функциональные тесты
1. Максимальная разница в середине вектора
Вход: {1, 10, 3, 10, 5};
Ожидаемый результат: (1, 10) (разница = 9).

2. Максимальная разница в начале
Вход: {5, 20, 15, 10};
Ожидаемый результат: (5, 20) (разница = 15).

3. Вектор из одного элемента
Вход: {7};
Ожидаемый результат: (7, 7) (разница = 0).

4. Максимальная разница в конце
Вход: {0, 100, 100, 25};
Ожидаемый результат: (0, 100) (разница = 100).

5. Отрицательные числа
Вход: {-10, -5, -1, 3, 0, 0};
Ожидаемый результат: (-10, -5) (разница = 5).

### 6.3 Тесты производительности

Генерация тестовых данных:
- Размер вектора: 30 000 000 элементов;
- Диапазон значений: [-100000, 100000].

Используется генератор псевдослучайных чисел std::mt19937

Максимальная пара создается в случайной позиции для тестирования разных сценариев положения пары элементов. Разница гарантированно максимальная: 300000 (превышает диапазон случайных значений). Остальные элементы заполняются случайными значениями.

## 7. Результаты и обсуждение

### 7.1 Проверка корректности
Корректность работы алгоритмов проверялась с помощью:
- Функциональных тестов с заранее известными результатами
- Сравнения результатов последовательной и параллельной версий (тесты с одинаковыми данными запускались для обеих версий реализации)
- Тестов на граничные случаи (пустой вектор, один элемент)

### 7.2 Performance
Были запущены тесты производительности с различными входными данными. (n - размер вектора) Результаты представлены в таблицах:

1. n = 10 000 000

|Режим	|Процессы |Время, с	|Ускорение  |Эффективность|
|-------|---------|---------|-----------|-------------|
|seq	  |1	      |0.00867  |1.00	      |N/A          |
|mpi	  |2	      |0.01478  |0.59       |29.5%        |
|mpi	  |4	      |0.02194  |0.40       |10.0%        |
|mpi	  |8	      |0.04191  |0.21	      |2.6%         |

2. n = 30 000 000

|Режим |Процессы |Время, с |Ускорение |Эффективность|
|------|---------|---------|----------|-------------|
|seq   |1        |0.02383  |1.00      |N/A          |
|mpi   |2        |0.06083  |0.39	    |19.5%        |
|mpi   |4        |0.13068  |0.18      |4.5%         |
|mpi   |8        |0.18740  |0.13      |1.6%         |

Ускорение вычислялось по формуле (seq_time / mpi_time)
Эффективность вычислялась по формуле (Ускорение / N) * 100%, где N - количество процессов

### 7.3 Анализ результатов
MPI версия демонстрирует замедление по сравнению с последовательной реализацией. Лучшая эффективность достигалась на 2 процессах: 19.5-29.5%. Эффективность падает с ростом количества процессов: на 8 процессах эффективность падает до 1.6-2.6%.

Для большего размера вектора (30M) эффективность стала заметно ниже.

Возможные причины низкой производительности:
- Высокие накладные расходы на распределение данных и сбор результатов перекрывают выгоду от параллельного исполнения;
- Относительно небольшая вычислительная линейная сложность алгоритма O(n);
- Нехватка свободных ядер для вычислений;
- Нехватка оперативной памяти.

Основные узкие места производительности:
- Распределение данных через последовательные MPI_Send/MPI_Recv;
- Сбор результатов через MPI_Gather (3 отдельных вызова);
- Рассылка результата через 2 отдельных MPI_Bcast.

## 8. Conclusions
Был разработан алгоритм для нахождения наиболее отличающихся по значению соседних элементов вектора. Реализованные последовательная и параллельная MPI-реализации успешно проходят все функциональные тесты, включая обработку граничных случаев и работу с отрицательными числами.

MPI-реализация демонстрирует отрицательное ускорение по сравнению с последовательной версией. Наилучшая эффективность достигается при использовании 2 процессов для вычисления. С ростом количества процессов и объемов данных эффективность значительно снижается - операции коммуникации между процессами достаточно дорогостоящие - затраты на распределение данных и сбор результатов превышают выгоду от параллельных вычислений.

Вывод: алгоритм не демонстрирует хорошей масштабируемости при увеличении числа процессов или объема данных. Алгоритм имеет линейную сложность O(n), что делает его менее подходящим для распараллеливания по сравнению с более сложными алгоритмами.

## 9. References

Open MPI Documentation. https://www.open-mpi.org/doc/
Статья "Основы MPI". https://habr.com/ru/articles/121925/
Учебное пособие "MPI для начинающих". https://parallel.uran.ru/node/182
Вспомогательные советы. http://stackoverflow.com/

## Приложение
```cpp
bool LeonovaAMostDiffNeighVecElemsMPI::RunImpl() {
  const auto &input_vec = GetInput();

  if (!ValidationImpl()) {
    return false;
  }

  if (input_vec.size() == 1) {
    GetOutput() = std::make_tuple(input_vec[0], input_vec[0]);
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = static_cast<int>(input_vec.size());
  if (size > 1) {
    ProcessWithMultipleProcesses(rank, size, total_size, input_vec);
  } else {
    ProcessSequentially(input_vec);
  }

  return true;
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessWithMultipleProcesses(int rank, int size, int total_size,
                                                                    const std::vector<int> &input_vec) {
  int actual_processes = std::min(size, total_size);
  int local_max_diff = -1;
  int local_first = 0;
  int local_second = 0;

  if (rank < actual_processes) {
    ProcessLocalData(rank, actual_processes, total_size, input_vec, local_max_diff, local_first, local_second);
  }

  GatherAndProcessResults(rank, actual_processes, local_max_diff, local_first, local_second, size);
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessLocalData(int rank, int actual_processes, int total_size,
                                                        const std::vector<int> &input_vec, int &local_max_diff,
                                                        int &local_first, int &local_second) {
  int chunk_size = total_size / actual_processes;
  int remainder = total_size % actual_processes;

  int my_size = chunk_size + (rank < remainder ? 1 : 0) + 1;
  int my_offset = (rank * chunk_size) + std::min(rank, remainder);

  if (rank == actual_processes - 1) {
    my_size = total_size - my_offset;
  }

  std::vector<int> local_data(my_size);
  ReceiveLocalData(rank, actual_processes, input_vec, my_size, local_data, total_size);

  FindLocalMaxDiff(local_data, local_max_diff, local_first, local_second);
}

void LeonovaAMostDiffNeighVecElemsMPI::ReceiveLocalData(int rank, int actual_processes,
                                                        const std::vector<int> &input_vec, int my_size,
                                                        std::vector<int> &local_data, int total_size) {
  if (rank == 0) {
    for (int index = 0; index < my_size; ++index) {
      local_data[index] = input_vec[index];
    }

    for (int dest = 1; dest < actual_processes; ++dest) {
      SendDataToProcess(dest, actual_processes, input_vec, total_size);
    }
  } else {
    MPI_Recv(local_data.data(), my_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::SendDataToProcess(int dest, int actual_processes,
                                                         const std::vector<int> &input_vec, int total_size) {
  int chunk_size = total_size / actual_processes;
  int remainder = total_size % actual_processes;

  int dest_size = chunk_size + (dest < remainder ? 1 : 0) + 1;
  int dest_offset = (dest * chunk_size) + std::min(dest, remainder);

  if (dest == actual_processes - 1) {
    dest_size = total_size - dest_offset;
  }

  MPI_Send(input_vec.data() + dest_offset, dest_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
}

void LeonovaAMostDiffNeighVecElemsMPI::FindLocalMaxDiff(const std::vector<int> &local_data, int &local_max_diff,
                                                        int &local_first, int &local_second) {
  for (int index = 0; index < static_cast<int>(local_data.size()) - 1; ++index) {
    int diff = std::abs(local_data[index] - local_data[index + 1]);
    if (diff > local_max_diff) {
      local_max_diff = diff;
      local_first = local_data[index];
      local_second = local_data[index + 1];
    }
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::GatherAndProcessResults(int rank, int actual_processes, int local_max_diff,
                                                               int local_first, int local_second, int size) {
  if (rank >= actual_processes) {
    local_max_diff = -1;
    local_first = 0;
    local_second = 0;
  }

  std::vector<int> all_diffs(size);
  std::vector<int> all_firsts(size);
  std::vector<int> all_seconds(size);

  MPI_Gather(&local_max_diff, 1, MPI_INT, all_diffs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_first, 1, MPI_INT, all_firsts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_second, 1, MPI_INT, all_seconds.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    FindGlobalMaxDiff(all_diffs, all_firsts, all_seconds, actual_processes);
  }

  BroadcastResult(rank);
}

void LeonovaAMostDiffNeighVecElemsMPI::FindGlobalMaxDiff(const std::vector<int> &all_diffs,
                                                         const std::vector<int> &all_firsts,
                                                         const std::vector<int> &all_seconds, int actual_processes) {
  int global_max_diff = -1;
  int global_first = 0;
  int global_second = 0;

  for (int index = 0; index < actual_processes; ++index) {
    if (all_diffs[index] > global_max_diff) {
      global_max_diff = all_diffs[index];
      global_first = all_firsts[index];
      global_second = all_seconds[index];
    }
  }

  GetOutput() = std::make_tuple(global_first, global_second);
}

void LeonovaAMostDiffNeighVecElemsMPI::BroadcastResult(int rank) {
  int result_first = std::get<0>(GetOutput());
  int result_second = std::get<1>(GetOutput());

  MPI_Bcast(&result_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_second, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = std::make_tuple(result_first, result_second);
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessSequentially(const std::vector<int> &input_vec) {
  int max_diff = -1;
  std::tuple<int, int> best_pair(0, 0);

  for (std::size_t index = 0; index < input_vec.size() - 1; ++index) {
    int diff = std::abs(input_vec[index] - input_vec[index + 1]);
    if (diff > max_diff) {
      max_diff = diff;
      best_pair = std::make_tuple(input_vec[index], input_vec[index + 1]);
    }
  }

  GetOutput() = best_pair;
}