# Скалярное произведение векторов

- Student: Завьялов Алексей Алексеевич, group 3823Б1ФИ3
- Technology: SEQ | MPI
- Variant: 9

## 1. Introduction


- Brief motivation, problem context, expected outcome.**

## 2. Problem Statement
Скалярное произведение двух векторов одной размерности - это число, равное сумме произведений соответствующих координат этих векторов.
Требуется найти скалярное произведение двух заданных векторов.

Формат входных данных:
Два математических вектора произвольной размерности, состоящих из действительных чисел.

Формат выходных данных:
Одно действительное число - скалярное произведение заданных векторов.

- Formal task definition, input/output format, constraints.**

## 3. Baseline Algorithm (Sequential)
Инициализируем результат значением 0. Затем последовательно проходим по векторам, добавляя к результату произведение соответствующих координат.

- Describe the base algorithm with enough detail to reproduce.

## 4. Parallelization Scheme
MPI:
Делим размер одного из векторов нацело на n, где n - число процессов, получаем размер блока - обозначим как block_size.
Далее каждый процесс вычисляет block_size слагаемых, составляющих скалярное произведение.


- For MPI: data distribution, communication pattern/topology, rank roles.
- For threads: decomposition, scheduling, synchronization.
Diagrams or short pseudocode are welcome.

## 5. Implementation Details
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

## 9. References
1. <Article/Book/Doc URL>
2. <Another source>

## Appendix (Optional)
```cpp
// Short, readable code excerpts if needed
```