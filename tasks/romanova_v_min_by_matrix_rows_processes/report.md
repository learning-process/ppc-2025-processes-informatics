# Нахождение минимальных значений по строкам матрицы

- Студент: Романова Василиса Дмитриевна, группа 3823Б1ФИ3
- Технология: SEQ, MPI
- Вариант: 17

## 1. Введение
- Мотивация: 
- Проблема: 
- Ожидаемый результат: 

## 2. Постановка задачи
Formal task definition, input/output format, constraints.

## 3. Базовый алгоритм (последовательный)
Describe the base algorithm with enough detail to reproduce.

## 4. Описание параллельного алгоритма
data distribution, communication pattern/topology, rank roles.

## 5. Experimental Setup
- Hardware/OS: CPU model, cores/threads, RAM, OS version
- Toolchain: compiler, version, build type (Release/RelWithDebInfo)
- Environment: PPC_NUM_THREADS / PPC_NUM_PROC, other relevant vars
- Data: how test data is generated or sourced (relative paths)

## 6. Результаты

### 6.1 Корректность


### 6.2 Производительность
| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.234   | 1.00    | N/A        |
| omp         | 2     | 0.700   | 1.76    | 88.0%      |
| omp         | 4     | 0.390   | 3.16    | 79.0%      |

## 7. Выводы


## 8. Литература
1. Стандарт MPI.
2. Лекции и практики по параллельному программированию.

## 9. Приложение

```cpp
int main(){ return 0; }
```