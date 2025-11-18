# Интегрирование – метод прямоугольников

- Студент: Романов Артем Сергеевич, группа 3823Б1ФИ3
- Технологии: SEQ, MPI
- Вариант: 19 

## 1. Введение
Интегрирование функций является составной частью многих научных и технических задач. Поскольку аналитическое интегрирование не всегда возможно, часто используются различные методы численного интегрирования. Метод прямоугольников является одним из наиболее простых для понимания и реализации методов, идея которого заключается в приближении значения интеграла некоторой интегральной суммой.

В рамках данной работы реализуется последовательная и параллельная версия алгоритма численного вычисления значения интеграла методом прямоугольников, а также исследуется эффективность их работы.

## 2. Постановка задачи
Пусть задана функция $f(x): \mathbb{R} \to \mathbb{R},$ которая  интегрируема на заданном отрезке $[a,b]$. Требуется приближённо вычислить значение определённого интеграла

$I = \int\limits_a^b f(x)dx$

с помощью метода прямоугольников.

Входными данными являются функция $f(x): \mathbb{R} \to \mathbb{R}$, вещественные границы отрезка интегрирования $a, b$  $(a < b)$, на котором функция определена и число слагаемых $n$ в интегральной сумме.


## 3. Базовый алгоритм (последовательный)
Метода прямоугольников заключается приближении значения определённого интеграла с помощью интегральной суммы:

$\int_{a}^{b} f(x) dx \approx \sum_{i=1}^{n} f(x_{i}^{\*}) \cdot \Delta x_{i}$,где  $x_{i}^{\*} \in [x_{i-1}, x_{i}]$ и $\Delta x_i = x_i - x_{i-1}$

Все $\Delta x_i$ полагаются равными $\frac{b-a}{n}$, где $n$ - число элементарных отрезков, а $x_i^*$ выбирается по-разному в зависимости от варианта метода прямоугольников. Основных вариаций три:
1. **Метод левых прямоугольников** - используется значение функции в левой границе каждого элементарного отрезка: $x_i^* = x_{i-1}$;  
2. **Метод правых прямоугольников** - используется значение функции в правой границе каждого элементарного отрезка: $x_i^* = x_{i}$;  
3. **Метод средних прямоугольников** - используется значение функции в середине каждого элементарного отрезка: $x_i^* = \frac{x_{i-1} + x_{i}}{2}$.  

В данной работе реализуется метод средних прямоугольников.
## 4. Схема паралеллизации
- For MPI: data distribution, communication pattern/topology, rank roles.
- For threads: decomposition, scheduling, synchronization.
Diagrams or short pseudocode are welcome.

## 5. Детали реализации
- Code structure (files, key classes/functions)
- Important assumptions and corner cases
- Memory usage considerations

## 6. Условия проведения экспериментов
- Hardware/OS: CPU model, cores/threads, RAM, OS version
- Toolchain: compiler, version, build type (Release/RelWithDebInfo)
- Environment: PPC_NUM_THREADS / PPC_NUM_PROC, other relevant vars
- Data: how test data is generated or sourced (relative paths)

## 7. Результаты

### 7.1 Корректность
Briefly explain how correctness was verified (reference results, invariants, unit tests).

### 7.2 Эффективность
Present time, speedup and efficiency. Example table:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.234   | 1.00    | N/A        |
| omp         | 2     | 0.700   | 1.76    | 88.0%      |
| omp         | 4     | 0.390   | 3.16    | 79.0%      |

Optionally add plots (use relative paths), and discuss bottlenecks and scalability limits.

## 8. Заключение
Summarize findings and limitations.

## 9. Список литературы
1. Павлова Т. Ю. Численное интегрирование [Электронный ресурс]. – Кемерово: Кемеровский государственный университет, Институт фундаментальных наук. – Режим доступа: [https://ifn.kemsu.ru/page_teachers/pavlova/Numerical_integration.Brief_theory.pdf](https://ifn.kemsu.ru/page_teachers/pavlova/Numerical_integration.Brief_theory.pdf) – (дата обращения: 12.11.2025).


## Приложение
```cpp
// Short, readable code excerpts if needed
```