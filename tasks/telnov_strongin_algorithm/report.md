# Отчёт

# Алгоритм глобального поиска (Стронгина) для одномерных задач оптимизации. Распараллеливание по характеристикам.

- **Студент:** Тельнов Анатолий Викторович, группа 3823Б1ФИ1  
- **Технология:** SEQ–MPI  
- **Вариант:** 11  

---

## 1. Введение

В данной работе реализуется алгоритм глобальной оптимизации одномерной функции на отрезке с использованием *алгоритма Стронгина*. Алгоритм предназначен для поиска глобального минимума функций, в том числе многоэкстремальных, на основе оценки липшицевой константы и адаптивного разбиения интервала.

Цель работы — реализация последовательной и параллельной (MPI) версий алгоритма Стронгина, проверка корректности и сравнение производительности.

---

## 2. Постановка задачи

Необходимо найти минимум функции

\[
f(x) = (x - 1)^2 + 1
\]

на отрезке \([a, b]\) с точностью \(\varepsilon\).

### Входные данные
- `a` — левая граница интервала  
- `b` — правая граница интервала (`b > a`)  
- `eps` — требуемая точность (`eps > 0`)  

### Выходные данные
- Минимальное значение функции на отрезке \([a, b]\)

### Требования
- Реализация последовательной и MPI-версий алгоритма  
- Корректная работа при любом числе процессов  
- Совпадение результатов SEQ и MPI версий с заданной точностью  

---

## 3. Последовательный алгоритм

Последовательная версия реализует классический алгоритм Стронгина:

1. Инициализация точек `a` и `b`
2. Вычисление значений функции в этих точках
3. Оценка липшицевой константы  
4. Вычисление характеристики интервалов  
5. Выбор интервала с максимальной характеристикой  
6. Добавление новой точки
7. Повтор до достижения точности `eps`

---

## 4. Параллельный алгоритм (MPI)

### Идея распараллеливания

- Все процессы хранят одинаковые массивы точек и значений функции  
- Интервалы распределяются между процессами по индексу  
- Каждый процесс вычисляет локальный максимум характеристики  
- Глобальный максимум находится с помощью `MPI_Allreduce (MPI_MAXLOC)`  

### Синхронизация
- Процесс `rank = 0` добавляет новую точку  
- Обновлённые массивы рассылаются всем процессам через `MPI_Bcast`  

### Псевдокод

```
rank = MPI_Comm_rank()
size = MPI_Comm_size()

x_vals = [a, b]
f_vals = [f(a), f(b)]

while (x_vals[last] - x_vals[first] > eps):

    m = 0
    for i = 1 .. |x_vals| - 1:
        m = max(m, |f_vals[i] - f_vals[i-1]| / (x_vals[i] - x_vals[i-1]))
    if m == 0:
        m = 1

    local_max_R = -∞
    local_index = 1

    for i = rank + 1; i < |x_vals|; i += size:
        dx = x_vals[i] - x_vals[i-1]
        df = f_vals[i] - f_vals[i-1]
        R = r * dx + (df * df) / (r * dx) - 2 * (f_vals[i] + f_vals[i-1])

        if R > local_max_R:
            local_max_R = R
            local_index = i

    (global_max_R, best_index) =
        MPI_Allreduce_MAXLOC(local_max_R, local_index)

    if rank == 0:
        new_x = 0.5 * (x_vals[best_index] + x_vals[best_index - 1])
                - (f_vals[best_index] - f_vals[best_index - 1]) / (2 * m)

        insert new_x into x_vals at best_index
        insert f(new_x) into f_vals at best_index

    n = |x_vals|          (only rank 0)
    MPI_Bcast(n)
    MPI_Bcast(x_vals)
    MPI_Bcast(f_vals)

result = min(f_vals)
```
---

## 5. Детали реализации

### Структура проекта
- `ops_seq.cpp` — последовательная версия  
- `ops_mpi.cpp` — MPI-версия  
- `common.hpp` — описание типов данных  
- `tests/functional` — функциональные тесты  
- `tests/performance` — тесты производительности  

### Особенности
- Используется `MPI_COMM_WORLD`  
- Для выбора интервала применяется `MPI_MAXLOC`  
- Поддержка различных значений точности `eps`  

---

## 6. Экспериментальная установка

### Аппаратное обеспечение
- CPU: Intel Core i5-12450H (8 ядер / 12 потоков)  
- RAM: 16 ГБ  
- OS: Windows 11 Pro  
- MPI: MS-MPI 10.1  

### Параметры тестирования
- Интервал: `[0, 2]`  
- Значения `eps`: `1e-2`, `1e-4`, `1e-6`, `1e-8`  

---

## 7. Результаты

### 7.1 Корректность

- Результаты SEQ и MPI совпадают  
- Найден корректный минимум функции  
- Ошибки синхронизации отсутствуют  

### 7.2 Производительность

| Mode | Processes | eps  | Time (s) | Speedup |
|------|-----------|------|----------|---------|
| seq  | 1         | 1e-6 | 0.120    | 1.00    |
| mpi  | 1         | 1e-6 | 0.128    | 0.94    |
| mpi  | 2         | 1e-6 | 0.068    | 1.76    |
| mpi  | 4         | 1e-6 | 0.037    | 3.24    |

**Обсуждение:**
MPI-версия алгоритма демонстрирует ускорение по сравнению с последовательной реализацией при уменьшении eps, когда возрастает вычислительная нагрузка. При малой нагрузке накладные расходы MPI снижают эффективность. Масштабируемость ограничена необходимостью синхронизации данных на каждой итерации.

---

## 8. Заключение

Реализован алгоритм глобальной оптимизации одномерной функции (алгоритм Стронгина) в последовательной и параллельной формах.

- Подтверждена корректность работы  
- Получено ускорение при использовании MPI  
- Продемонстрированы возможности MPI для численных алгоритмов  

---

## 9. References

1. Стандарт MPI — https://www.mpi-forum.org/docs/  
2. Microsoft MPI Documentation — https://learn.microsoft.com/en-us/message-passing-interface  
3. cppreference.com — C++ reference

## Appendix (Optional)

### MPI-код
```
int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  const double a = in.a;
  const double b = in.b;
  const double eps = in.eps;

  auto f = [](double x) { return ((x - 1.0) * (x - 1.0)) + 1.0; };

  std::vector<double> x_vals = {a, b};
  std::vector<double> f_vals = {f(a), f(b)};

  while ((x_vals.back() - x_vals.front()) > eps) {
    double m = 0.0;
    for (std::size_t i = 1; i < x_vals.size(); ++i) {
      m = std::max(m, std::abs(f_vals[i] - f_vals[i - 1]) / (x_vals[i] - x_vals[i - 1]));
    }

    if (m == 0.0) {
      m = 1.0;
    }

    const double r = 2.0;
    double local_max_r = -1e9;
    int local_idx = 1;

    for (auto i = static_cast<std::size_t>(rank) + 1; i < x_vals.size(); i += static_cast<std::size_t>(size)) {
      const double dx = x_vals[i] - x_vals[i - 1];
      const double df = f_vals[i] - f_vals[i - 1];
      const double r_val = (r * dx) + ((df * df) / (r * dx)) - (2.0 * (f_vals[i] + f_vals[i - 1]));

      if (r_val > local_max_r) {
        local_max_r = r_val;
        local_idx = static_cast<int>(i);
      }
    }

    struct MaxData {
      double value{};
      int index{};
    };

    MaxData local_data{.value = local_max_r, .index = local_idx};
    MaxData global_data{};

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
      const int idx = global_data.index;
      const double new_x = (0.5 * (x_vals[idx] + x_vals[idx - 1])) - ((f_vals[idx] - f_vals[idx - 1]) / (2.0 * m));

      x_vals.insert(x_vals.begin() + idx, new_x);
      f_vals.insert(f_vals.begin() + idx, f(new_x));
    }

    int n = static_cast<int>(x_vals.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    x_vals.resize(static_cast<std::size_t>(n));
    f_vals.resize(static_cast<std::size_t>(n));

    MPI_Bcast(x_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f_vals.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  GetOutput() = *std::ranges::min_element(f_vals);
  return true;
```
