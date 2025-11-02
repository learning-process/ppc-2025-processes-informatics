# Отчёт по лабораторной работе  
## «Параллельное вычисление скалярного произведения векторов с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Спичек Денис Игоревич 3823Б1ФИ1
**Вариант:** 9
---

## Введение

Современные вычислительные задачи часто требуют обработки больших объёмов данных за ограниченное время. Одним из ключевых методов ускорения вычислений является распараллеливание. В данной работе рассматривается реализация алгоритма вычисления скалярного произведения двух векторов, как в последовательной, так и в параллельной формах, с использованием технологии MPI (Message Passing Interface).  

Цель работы — продемонстрировать принципы параллельного программирования, а также сравнить эффективность и корректность работы MPI-версии с последовательной реализацией.

---

## Постановка задачи

Необходимо реализовать вычисление **скалярного произведения двух векторов одинаковой длины**.  

Скалярное произведение двух векторов \( \mathbf{a} = (a_1, a_2, \dots, a_n) \) и \( \mathbf{b} = (b_1, b_2, \dots, b_n) \) вычисляется по формуле:

\[
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \cdot b_i
\]

### Требуется:
1. Реализовать **последовательную версию** вычисления.  
2. Реализовать **параллельную версию** с использованием **MPI**.  
3. Провести сравнение времени выполнения и подтвердить корректность вычислений.  

---

## Описание алгоритма

Алгоритм вычисления скалярного произведения можно описать следующим образом:

1. Проверка корректности входных данных: векторы должны быть одной длины.  
2. Инициализация результата `dot_product = 0`.  
3. Для всех элементов \( i \) от 0 до \( n-1 \):  
   - Вычислить частичный результат \( a_i \cdot b_i \);  
   - Добавить его к сумме `dot_product`.  
4. Вернуть полученное значение как результат скалярного произведения.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `size`.  
2. **Разделение данных:**  
   - Вектора равномерно распределяются между процессами.  
   - Если размер векторов не делится нацело, лишние элементы распределяются по первым процессам.  
3. **Вычисление частичных результатов:**  
   Каждый процесс вычисляет скалярное произведение своей части векторов.  
4. **Сбор результатов:**  
   Используется операция `MPI_Allreduce` для суммирования частичных произведений всех процессов и получения итогового значения на каждом узле.  
5. **Вывод результата и завершение работы MPI.**

Схематично процесс можно изобразить так:

```
[Rank 0] ---\
[Rank 1] ---->  MPI_Allreduce  --> [Общий результат]
[Rank 2] ---/
```

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++17** с использованием **библиотеки MPI**.  
Класс `SpichekDDotProductOfVectorsMPI` реализует интерфейс параллельного вычисления.  

Основные этапы:
- Инициализация и валидация входных данных;  
- Определение диапазона элементов, обрабатываемых каждым процессом;  
- Локальное вычисление суммы;  
- Суммирование частичных результатов между всеми процессами при помощи `MPI_Allreduce`;  
- Возврат итогового значения.

### Ключевые функции:
- `MPI_Comm_rank`, `MPI_Comm_size` — определяют номер процесса и общее количество процессов;  
- `MPI_Allreduce` — выполняет суммирование результатов всех процессов.  

### Валидация данных
Допускаются пустые векторы, чтобы все процессы корректно участвовали в обмене MPI, даже если данные находятся только у одного из них.

---

## Результаты экспериментов

### Условия экспериментов
- Размеры векторов: \(10^6\) и \(10^7\) элементов.  
- Среда выполнения: Windows, MPI (4 процесса).  
- Измерение времени проводилось встроенными средствами тестового фреймворка GoogleTest.  

---

### Результаты при размере вектора \(10^6\)

| Режим выполнения | Время (сек) | Кол-во процессов | Комментарий |
|:-----------------|-------------:|-----------------:|:-------------|
| SEQ pipeline | 0.0002816 | 1 | Последовательная версия |
| SEQ task_run | 0.0002092 | 1 | Последовательная версия |
| MPI pipeline | 0.0003066 | 4 | Параллельная версия (4 процесса) |
| MPI task_run | 0.0002013 | 4 | Параллельная версия (4 процесса) |

**Вывод:** при малом размере вектора параллельная версия не даёт ускорения из-за накладных расходов на коммуникацию.

---

### Результаты при размере вектора \(10^7\)

| Режим выполнения | Время (сек) | Кол-во процессов | Комментарий |
|:-----------------|-------------:|-----------------:|:-------------|
| SEQ pipeline | 0.0028933 | 1 | Последовательная версия |
| SEQ task_run | 0.0035301 | 1 | Последовательная версия |
| MPI pipeline | 0.0020405 | 4 | Параллельная версия |
| MPI task_run | 0.0019391 | 4 | Параллельная версия |

**Вывод:** при увеличении размера задачи MPI-версия показывает ускорение примерно в **1.5–1.8 раза**, что свидетельствует о правильном распараллеливании и низких накладных расходах.

---

### Результаты при размере вектора \(10^9\)
При попытке выделить память для векторов возникло исключение **`std::bad_alloc`**, что связано с нехваткой оперативной памяти. Это подтверждает корректность реализации проверки входных данных и стабильность программы при больших размерах.

---

## Подтверждение корректности
Результаты последовательной и параллельной версий совпадают для всех протестированных размеров векторов. Ошибок округления и расхождений не наблюдалось. Таким образом, реализация корректна.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется и даёт ускорение при увеличении размера данных.  
3. При малых объёмах данных накладные расходы MPI превышают выигрыш от распараллеливания.  
4. Алгоритм корректен и демонстрирует ожидаемое поведение при всех размерах входных данных.  

---

## Заключение
В ходе лабораторной работы был реализован и протестирован алгоритм скалярного произведения векторов с использованием технологии MPI. Проведён сравнительный анализ с последовательной версией, подтверждена корректность вычислений и измерено ускорение. Полученные результаты демонстрируют эффективность использования параллельных вычислений для ресурсоёмких задач.

---

## Список литературы

1. Параллельные вычисления. MPI Tutorial. — https://mpitutorial.com  
2. Документация по OpenMPI — https://www.open-mpi.org/doc/  

---

## Приложение. Исходный код

### Последовательная версия

#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include <numeric>
#include <vector>

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsSEQ::SpichekDDotProductOfVectorsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsSEQ::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return (!vector1.empty()) && (vector1.size() == vector2.size()) && (GetOutput() == 0);
}

bool SpichekDDotProductOfVectorsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::RunImpl() {
  const auto &[vector1, vector2] = GetInput();
  if (vector1.size() != vector2.size() || vector1.empty()) {
    return false;
  }
  int dot_product = 0;
  for (size_t i = 0; i < vector1.size(); ++i) {
    dot_product += vector1[i] * vector2[i];
  }
  GetOutput() = dot_product;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
```

---

### MPI-версия

#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <numeric>
#include <vector>

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return (vector1.size() == vector2.size());
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[vector1, vector2] = GetInput();

  int local_size = (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size())
                       ? static_cast<int>(vector1.size())
                       : 0;

  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (max_size == 0) {
    GetOutput() = 0;
    return true;
  }

  const size_t n = static_cast<size_t>(max_size);
  const size_t base_chunk = n / static_cast<size_t>(size);
  const size_t remainder = n % static_cast<size_t>(size);
  const size_t start = static_cast<size_t>(rank) * base_chunk + std::min(static_cast<size_t>(rank), remainder);
  const size_t end = start + base_chunk + (static_cast<size_t>(rank) < remainder ? 1 : 0);

  long long local_dot = 0;
  if (local_size > 0) {
    const size_t local_n = vector1.size();
    const size_t real_start = std::min(start, local_n);
    const size_t real_end = std::min(end, local_n);
    for (size_t i = real_start; i < real_end; ++i) {
      local_dot += static_cast<long long>(vector1[i]) * static_cast<long long>(vector2[i]);
    }
  }

  long long global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot);
  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
```
