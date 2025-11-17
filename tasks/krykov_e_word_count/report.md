# Отчёт по лабораторной работе  
## «Параллельный подсчет числа слов в строке с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Крюков Егор Федорович 3823Б1ФИ1
**Вариант:** 24


## Введение

Подсчет слов в строке - базовая операция, используемая в текстовых анализаторах, поисковых системах и системах обработки естественного языка. Параллельные вычисления позволяют ускорить обработку больших объемов текстовых данных, что особенно актуально в эпоху больших данных.
В рамках данной работы был реализован алгоритм подсчета числа слов: последовательная и параллельная реализация.


## Постановка задачи

Задача заключается в определении количества слов в заданной текстовой строке. Входными данными является строка текста типа std::string, выходными - целое число, представляющее количество слов. Под словом понимается последовательность непробельных символов, ограниченная пробельными символами. К пробельным символам относятся пробел, табуляция и перевод строки. Алгоритм должен корректно обрабатывать различные граничные случаи, включая наличие ведущих и завершающих пробелов, множественных пробелов между словами, а также строки, содержащие различные комбинации пробельных символов.


## Описание алгоритма

Базовый последовательный алгоритм реализован в классе KrykovEStringWordCountSEQ. Алгоритм начинает работу с инициализации состояния in_word = false и счетчика word_count = 0. Затем осуществляется последовательное чтение каждого символа входной строки. При обнаружении непробельного символа, если текущее состояние указывает на нахождение вне слова, происходит увеличение счетчика и установка флага in_word в true. При встрече пробельного символа флаг in_word сбрасывается. Важным этапом предобработки является удаление ведущих и завершающих пробельных символов, что обеспечивает корректность подсчета для строк с пробелами в начале или конце.


## Описание схемы параллельного алгоритма

Параллельная реализация с использованием MPI основывается на распределении исходной строки между доступными процессами. Процесс с рангом 0 выполняет роль координатора, передавая размер данные всем процессам. Каждый процесс независимо подсчитывает слова в своем блоке(чанке) данных, используя тот же алгоритм, что и последовательная версия. Для обеспечения корректности учета слов, пересекающих границы блоков, собирается информация о граничных символах каждого блока через MPI_Gather. Если блок заканчивается непробельным символом, а следующий блок начинается также с непробельного символа, осуществляется коррекция общего количества слов путем уменьшения счетчика. Финальный результат собирается на процессе с рангом 0 и распространяется среди всех процессов.


## Результаты экспериментов

Условия экспериментов
Размеры тестовой строки: 2*10^5, 2*10^6, 2*10^7, 2*10^8 
Среда выполнения: Windows, MPI (4 процесса).
Измерение времени — средствами тестового окружения GoogleTest (режимы pipeline и task_run).


### Результаты при размере тестовой строки 10^5

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00230948 | Параллельная версия     |
| MPI task_run     |  0.00131006 | Параллельная версия     |
| SEQ pipeline     |  0.00281244 | Последовательная версия |
| SEQ task_run     |  0.00364718 | Последовательная версия |

Вывод: уже на малом числе повторов MPI показывает небольшое преимущество, однако разница составляет лишь 10–20%. Накладные расходы MPI частично перекрывают преимущество параллельной реализации.


### Результаты при размере тестовой строки 10^6

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.02180310 | Параллельная версия     |
| MPI task_run     |  0.01254694 | Параллельная версия     |
| SEQ pipeline     |  0.03160660 | Последовательная версия |
| SEQ task_run     |  0.03092086 | Последовательная версия |

Вывод: при увеличении нагрузки параллельная версия становится быстрее в ~1.4–2.5 раза. Доля накладных расходов MPI становятся меньше.


### Результаты при размере тестовой строки 10^7

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.23840780 | Параллельная версия     |
| MPI task_run     |  0.11899546 | Параллельная версия     |
| SEQ pipeline     |  0.32742938 | Последовательная версия |
| SEQ task_run     |  0.30835360 | Последовательная версия |

Вывод: ускорение составляет примерно 1.3–2.6 раза. MPI начинает работать явно эффективнее SEQ версии.


### Результаты при размере тестовой строки 10^8

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  2.59652888 | Параллельная версия     |
| MPI task_run     |  1.23528820 | Параллельная версия     |
| SEQ pipeline     |  3.55990464 | Последовательная версия |
| SEQ task_run     |  3.34456572 | Последовательная версия |

Вывод: на больших размерах задачи ускорение MPI достигает ~1.3–2.7 раз, что демонстрирует эффективность алгоритма на больших объемах данных


## Заключение

В ходе работы успешно реализованы последовательный и параллельный алгоритмы подсчета слов в строке. Разработана эффективная схема распределения данных для MPI-версии с решением проблемы корректного учета слов на границах блоков. Установлено, что MPI-реализация имеет ограничения, связанные с накладными расходами на коммуникацию, что особенно заметно при обработке коротких строк. Эффективность параллельного алгоритма увеличивается с увеличением объема данных, так как накладные расходы становятся менее значительными на фоне основного алгоритма.


## Источники

1. Лекции Сысоева Александра Владимировича
2. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
3. Документация по курсу
4. Различные источники из интернета


## Приложение. Исходный код

### SEQ-версия

```cpp
#include "krykov_e_word_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "krykov_e_word_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krykov_e_word_count {

KrykovEWordCountSEQ::KrykovEWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  input.erase(input.begin(),
              std::find_if(input.begin(), input.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  input.erase(std::find_if(input.rbegin(), input.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              input.end());
  return true;
}

bool KrykovEWordCountSEQ::RunImpl() {
  const std::string &text = GetInput();
  if (text.empty()) {
    return false;
  }

  bool in_word = false;
  size_t word_count = 0;

  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (in_word) {
        in_word = false;
      }
    } else {
      if (!in_word) {
        in_word = true;
        word_count++;
      }
    }
  }

  GetOutput() = word_count;
  return true;
}

bool KrykovEWordCountSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count

```

### MPI-версия
```cpp
#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace krykov_e_word_count {

KrykovEWordCountMPI::KrykovEWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  auto trimmed = input;

  trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(),
                                              [](char ch) { return !std::isspace(static_cast<unsigned char>(ch)); }));

  trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(),
                             [](char ch) {
    return !std::isspace(static_cast<unsigned char>(ch));
  }).base(),
                trimmed.end());

  GetInput() = trimmed;
  return true;
}

bool KrykovEWordCountMPI::RunImpl() {
  const std::string &text = GetInput();
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (text.empty()) {
    GetOutput() = 0;
    return true;
  }

  int text_size = static_cast<int>(text.size());
  MPI_Bcast(&text_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_size = text_size / world_size;
  int remainder = text_size % world_size;

  std::vector<int> chunk_sizes(world_size);
  std::vector<int> displs(world_size);

  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    chunk_sizes[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += chunk_sizes[i];
  }

  int local_size = chunk_sizes[world_rank];
  std::vector<char> local_chunk(local_size);

  MPI_Scatterv(world_rank == 0 ? text.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_CHAR, local_chunk.data(),
               local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  size_t local_count = 0;
  bool in_word = false;
  for (char c : local_chunk) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      in_word = false;
    } else {
      if (!in_word) {
        in_word = true;
        local_count++;
      }
    }
  }

  int starts_with_space = local_size > 0 ? (std::isspace(static_cast<unsigned char>(local_chunk[0])) ? 1 : 0) : 1;
  int ends_with_space =
      local_size > 0 ? (std::isspace(static_cast<unsigned char>(local_chunk[local_size - 1])) ? 1 : 0) : 1;

  std::vector<int> all_starts(world_size);
  std::vector<int> all_ends(world_size);

  MPI_Gather(&starts_with_space, 1, MPI_INT, world_rank == 0 ? all_starts.data() : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&ends_with_space, 1, MPI_INT, world_rank == 0 ? all_ends.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<size_t> all_counts(world_size);
  MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG_LONG, world_rank == 0 ? all_counts.data() : nullptr, 1,
             MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    size_t total_count = 0;
    for (size_t count : all_counts) {
      total_count += count;
    }
    for (int i = 1; i < world_size; ++i) {
      if (all_ends[i - 1] == 0 && all_starts[i] == 0) {
        total_count--;
      }
    }
    GetOutput() = static_cast<int>(total_count);
  }

  int result = 0;
  if (world_rank == 0) {
    result = GetOutput();
  }
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank != 0) {
    GetOutput() = result;
  }

  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_word_count

```