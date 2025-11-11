# Интегрирование – метод прямоугольников

- **Студент:** Еремин Василий Егвеньевич, группа 3823Б1ФИ2
- **Технология:** SEQ | MPI
- **Вариант:** №19

## 1. Введение
Целью лабораторной было написание алгоритма интегрирования методом прямоугольников двумя разными способами: последовательный и через MPI

**Цель работы:** Реализация и сравнительный анализ последовательной и параллельной версий алгоритма численного интегрирования методом прямоугольников.

**Задачи:**
1. Реализовать последовательную версию алгоритма метода прямоугольников
2. Реализовать параллельную версию с использованием технологии MPI
3. Провести сравнительный анализ производительности и эффективности обеих реализаций

## 2. Постановка задачи

**Задача**: Требуется вычислить значение определенного интеграла методом прямоугольников

![Интеграл](data/int.jpeg)

где:
- `f(x)` — интегрируемая функция
- `a`, `b` — нижний и верхний пределы интегрирования соответственно

**Описание метода решения:** Из трех методов (левых, правых и средних прямоугольников) был выбран *метод средних прямоугольников*.

Область интегрирования разбивается на `n` равных отрезков. Сложная площадь под кривой разбивается на множество простых прямоугольников - на каждом отрезке строится прямоугольник высотой, равной значению функции в середине отрезка. Общая площадь вычисляется как сумма площадей всех этих прямоугольников.

![Метод средних прямоугольников](data/square.png) 

**Входные данные:**
- `a` — нижний предел интегрирования (вещественное число)
- `b` — верхний предел интегрирования (вещественное число)  
- `n` — количество отрезков разбиения (целое число)
- `f(x)` — интегрируемая функция

**Выходные данные:**
- Приближенное значение интеграла (вещественное число)

**Ограничения:** 
- `-1 000 000 000 ≤ a < b ≤ 1 000 000 000`
- `0 < n ≤ 100 000 000`


## 3. Описание алгоритма (gоследовательного)
Разделим наш отрезок интегрирования на n частей. Длину отвезка запишем как step_size((b-a) / n). Найдем площадь каждого квадрата по формуле f(a + (i + 0.5) * step_size); Сложим сумму площадей для получения резульатта.

**Алгоритм последовательного вычисления:**
1. Найдем шаг интегрирования `h = (b - a) / n`
2. Через цикл найдем площадь каждого квадрата по формуле `f(a + (i + 0.5) * h)` 
3. Сумму всех сначений умножим на шаг интегрирования

Код
```cpp
#include <iostream>
using namespace std;

double rectangle_method() {
    cout << "Hello World!" << endl;
    return 0;
}
```

**Математическое обоснование:** 
Формула метода средних прямоугольников

## 4. Схема распараллеливания (MPI)
- For MPI: data distribution, communication pattern/topology, rank roles.
- For threads: decomposition, scheduling, synchronization.
Diagrams or short pseudocode are welcome.

## 5. Экспериментальные результаты
- Code structure (files, key classes/functions)
- Important assumptions and corner cases
- Memory usage considerations

## 6. Experimental Setup
- Hardware/OS: CPU model, cores/threads, RAM, OS version
- Toolchain: compiler, version, build type (Release/RelWithDebInfo)
- Environment: PPC_NUM_THREADS / PPC_NUM_PROC, other relevant vars
- Data: how test data is generated or sourced (relative paths)

## 7. Results and Discussion

### 7.1 Correctness
Briefly explain how correctness was verified (reference results, invariants, unit tests).

### 7.2 Performance
Present time, speedup and efficiency. Example table:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.234   | 1.00    | N/A        |
| omp         | 2     | 0.700   | 1.76    | 88.0%      |
| omp         | 4     | 0.390   | 3.16    | 79.0%      |

Optionally add plots (use relative paths), and discuss bottlenecks and scalability limits.

## 8. Conclusions
Summarize findings and limitations.

## 9. Источники
1. [Wikipedia](https://ru.wikipedia.org/wiki/Метод_прямоугольников)
2. [Презентация по курсу](https://learning-process.github.io/parallel_programming_slides/slides/01-intro.pdf)

## Appendix (Optional)
```cpp
// Short, readable code excerpts if needed
```