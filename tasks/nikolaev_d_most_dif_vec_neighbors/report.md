# Нахождение наиболее отличающихся по значению соседних элементов вектора

- Студент: Николаев Денис Андреевич, группа 3823Б1ФИ1
- Технология: SEQ | MPI
- Вариант: 8

## 1. Введение
В данной лабораторной работе исследуется задача нахождение наиболее отличающихся по значению соседних элементов в векторе. Для решения данной задачи были реализованы 2 алгоритма - последовательный (SEQ) и параллельный (MPI).

## 2. Постановка задачи
Дан вектор из n целых чисел. Требуется найти пару соседних элементов с максимальной разницей между ними diff = abs(v_i+1 - v_i).

Формат входных данных - std::vector\<int> - входной вектор с целыми значениями, формат выходных данных - std::pair<int, int> - пара элементов из входного вектора с наибольшой разницей между ними.

Ограничения:
- Входной вектор должен иметь хотя бы 2 элемента внутри (1 пара)

## 3. Базовый алгоритм (Последовательный)
Последовательный алгоритм проходит по всем элементам входного вектора v от 0 до n-2 и высчитывает diff = abs(v[i+1] - v[i]) и сравнивает с max_diff. Если diff > max_diff, то сохраняем найденую пару (v[i], v[i+1]) в result_elements.

## 4. Схема распараллеливания
Входной вектор разбивается на непрерывные сегменты между процессами. Каждый процесс получает свой сегмент элементов (индексы start и end), при этом распределение осуществляется так, чтобы разница в количестве элементов на каждый процесс была не более одного.

Каждый процесс находит локальную пару с локальной максимальной разницей. Используется операция MPI_Allreduce с оператором MPI_MAXLOC, который находит максимальную разность среди всех процессов и сохраняет ранг процесса, в котором она была найдена.

Процесс, у которого была найдена максимальная разность, рассылает пару элементов (с максимальной разностью) всем остальным процессам с помощью MPI_Bcast. Это гарантирует, что все процессы получат правильный результат.

## 5. Детали реализации
Структура кода:

- ppc-2025-processes-informatics/tasks/nikolaev_d_most_dif_vec_neighbors/seq/src/ops_seq.cpp - последовательная реализация
- ppc-2025-processes-informatics/tasks/nikolaev_d_most_dif_vec_neighbors/mpi/src/ops_mpi.cpp - MPI реализация

Тесты:
- ppc-2025-processes-informatics/tasks/nikolaev_d_most_dif_vec_neighbors/tests/functional/main.cpp - функциональные
- ppc-2025-processes-informatics/tasks/nikolaev_d_most_dif_vec_neighbors/tests/performance/main.cpp - на производительность

Особенности:
- Код корректно обрабатывает случай, когда процессов больше, чем элементов, за счет корректировки end и последующей проверки (if start <= end). Лишние процессы ее не пройдут и их local_max_diff так и останется -1, а пара local_result будет равна {0, 0}, тоесть результаты этого "лишнего" процесса не будут влиять на результат.
- Разница в количестве элементов на каждый процесс не превышает одного. Первые `remainder` процессов получают по одному дополнительному элементу.
```cpp
int elements_per_process = n / size;  // базовое количество элементов на процесс
int remainder = n % size;            // остаточные элементы, которые мы распределим между процессами
int start = rank * elements_per_process + std::min(rank, remainder);
int end = start + elements_per_process - 1;
if (rank < remainder) {
    end++;
}
```

## 6. Эксперементальное окружение
### 6.1 Оборудование
- CPU: AMD Ryzen 5 2600 (6 cores, 12 threads, 3.4 GHz)
- RAM: 16 GB
- OS: Windows 10, сама программа была запущена через WSL2 на Ubuntu 24.04.2 LTS.
- Toolchain: GCC 13.3.0, Release build

### 6.2 Генерация данных
- Вектор из n = 100'000'000 целочисленных чисел
- Числа были сгенерированы с помощью генератора псевдослучайных чисел std::mt19937 с установленным сидом(42), чтобы между разными запусками сгенерированные числа оставались такими же
- Диапазон значений от -10'000 до 10'000

Подобным образом высчитывается случайная позиция max_dif_pos максимальной пары (индекс в пределах от 0 до n-2). На позициях max_dif_pos и max_dif_pos+1 встают числа -15'000 и 15'000 cоотвественно, что делает разницу между ними гарантировано максимальной.


## 7. Результаты и обсуждение

### 7.1 Проверка корректности
Корректность работы алгоритмов и их производительность проверялись с помощью:
- Функциональных тестов с заранее известными результатами
- Тесты производительности - для измерения производительности/времени выполнения работы алгоритмом

Обе версии алгоритма тестировались на идентичных наборах входных данных для обеспечения корректности сравнения.


### 7.2 Производительность
n = 1'000'000
| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.0921  | 1.00    | N/A        |
| mpi         | 2     | 0.0365  | 2.52    | 126.0%     |
| mpi         | 4     | 0.0317  | 2.91    | 72.8%      |
| mpi         | 8     | 0.0144  | 6.40    | 80.0%      |

n = 100'000'000
| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.0584  | 1.00    | N/A        |
| mpi         | 2     | 0.3741  | 2.83    | 141.5%     |
| mpi         | 4     | 0.2556  | 4.14    | 103.5%     |
| mpi         | 8     | 0.1209  | 8.75    | 109.4%     |

По результатам из таблицы наблюдается значительное ускорение параллельной версии относительно последовательной реализации. При использовании 8 процессов достигнуто ускорение в 8.75 раз.

При меньшем объеме данных увеличивается накладные расходы на коммуникации и синхронизацию.
Алгоритм эффективно масштабируется с увеличением числа процессов. При переходе от 2 к 8 процессами ускорение продолжает расти почти линейно.

Наилучшая эффективность для 100 миллионов элементов достигается при 2 процессах (141.5%), однако даже при 8 процессах эффективность остается высокой (109.4%), что свидетельствует о хорошей сбалансированности нагрузки и минимальных накладных расходах на коммуникации.

Наилучшая эффективность для 10 миллионов элементов достигается при 2 процессах (126%), что говорит о том, что для данного объема данных оптимально использовать меньше процессов.

При запуске с 8 процессами был использован флаг --oversubscribe, но даже не смотря на это эффективность осталась высокой.

## 8. Заключение
В рамках данной работы была успешно решена задача нахождение наиболее отличающихся по значению соседних элементов в векторе. Для решения данной задачи были успешно реализованы 2 алгоритма - последовательный (SEQ) и параллельный (MPI).

В итоге, MPI-версия показала значительное превосходство над последовательной реализацией. Чем больше данных, тем лучше себя показывает себя MPI-алгоритм, при этом сохраняя корректность результатов.

## 9. Список литературы
1. MPI Forum. **MPI: A Message-Passing Interface Standard. Version 3.1** [Электронный ресурс]. — Режим доступа: https://www.mpi-forum.org/docs/
2. Лекции по параллельному программированию
3. Практические занятия по параллельному программированию

## 10. Приложение (Optional)

## Реализация MPI алгоритма `ops_mpi.cpp`
```cpp
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

NikolaevDMostDifVecNeighborsMPI::NikolaevDMostDifVecNeighborsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool NikolaevDMostDifVecNeighborsMPI::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool NikolaevDMostDifVecNeighborsMPI::PreProcessingImpl() {
  return true;
}

bool NikolaevDMostDifVecNeighborsMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (input.size() < 2) {
    return false;
  }

  int n = input.size();
  int elements_per_process = n / size;
  int remainder = n % size;

  int start = rank * elements_per_process + std::min(rank, remainder);
  int end = start + elements_per_process - 1;
  if (rank < remainder) {
    end++;
  }

  if (rank == size - 1) {
    end = n - 2;
  } else {
    end = std::min(end, n - 2);
  }

  int local_max_diff = -1;
  std::pair<int, int> local_result = {0, 0};

  if (start <= end) {
    for (int i = start; i <= end; i++) {
      int diff = std::abs(input[i + 1] - input[i]);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_result = {input[i], input[i + 1]};
      }
    }
  }

  struct {
    int diff;
    int rank;
  } local_max, global_max;

  local_max.diff = local_max_diff;
  local_max.rank = rank;

  MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

  std::pair<int, int> global_result;
  if (rank == global_max.rank) {
    global_result = local_result;
  }

  MPI_Bcast(&global_result, 2, MPI_INT, global_max.rank, MPI_COMM_WORLD);

  output = global_result;
  return true;
}

bool NikolaevDMostDifVecNeighborsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_most_dif_vec_neighbors

```