# Отчёт по лабораторной работе
# Подсчёт числа буквенных символов в строке

- Student: Кичанова Ксения Константиновна, group 3823Б1ФИ3
- Technology: SEQ | MPI
- Variant: 22

## 1. Introduction
Подсчет буквенных символов в строке — классическая задача обработки текстовых данных. В контексте параллельного программирования данная задача представляет интерес благодаря возможности распараллеливания операции подсчета символов в больших текстовых данных. Ожидается, что при использовании технолоогии mpi произойдёт ускорение по сравнению с последовательной версией.

## 2. Problem Statement
Нужно найти количество буквенных символов в строке input_str.
Input: Строка (std::string)
Output: Целое число (int) - количество буквенных символов в строке

## 3. Baseline Algorithm (Sequential)
Базовый последовательный алгоритм реализует однопроходное сканирование входной строки. Алгоритм последовательно обрабатывает каждый символ строки, проверяя его принадлежность к множеству буквенных символов с помощью функции std::isalpha. Для каждого символа, удовлетворяющего условию, увеличивается счетчик.
Вычислительная сложность: O(N)

## 4. Parallelization Scheme
Каждый процесс MPI получает для обработки сегмент исходной строки. Размер сегмента вычисляется как целочисленное деление общей длины строки на количество процессов. Процесс с наибольшим рангом получает оставшуюся часть строки. Каждый процесс независимо выполняет подсчет буквенных символов в своем сегменте, проверяя каждый символ с помощью функции std::isalpha. После завершения локальных вычислений процессы синхронизируются: операция MPI_Reduce с операцией суммирования MPI_SUM собирает все локальные счетчики на процессе с рангом 0. Затем итоговый результат рассылается всем процессам с помощью MPI_Bcast для обеспечения идентичности выходных данных во всех процессах.

## 5. Implementation Details
- common.hpp - общие типы данных и константы
- ops_seq - последовательная реализация алгоритма
- ops_mpi - параллельная MPI-реализация алгоритма
- tests - functional для проверки корректности и performance для замера скорости.

## 6. Experimental Setup
- Аппаратное обеспечение: AMD Ryzen 5 5500U (6 ядер, 12 логических процессоров, базовая скорость 2,10 ГГц)
- ОЗУ — 8 ГБ 
- Операционная система: Windows 11
- Компилятор: g++
- Использовался Docker-контейнер.
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность реализации проверялась тестами, включающие пустые строки, цифры, знаки и длинные строки.

### 7.2 Performance

pipeline:

| Mode        | Count | Time, s   | Speedup | Efficiency |
|-------------|-------|-----------|---------|------------|
| seq         | 1     | 0.01322   | 1.00    | N/A        |
| omp         | 2     | 0.01044   | 1.27    | 63.5%      |
| omp         | 4     | 0.00748   | 1.77    | 44.3%      |

task_run:

| Mode        | Count | Time, s   | Speedup | Efficiency |
|-------------|-------|-----------|---------|------------|
| seq         | 1     | 0.01419   | 1.00    | N/A        |
| omp         | 2     | 0.00865   | 1.64    | 82.0%      |
| omp         | 4     | 0.00569   | 2.49    | 62.3%      |


## 8. Conclusions
MPI реализация демонстрирует положительное ускорение на всех конфигурациях. Максимальное ускорение 2.49 достигнуто в режиме task_run на 4 процессах. Эффективность параллелизации снижается с ростом числа процессов из-за накладных расходов MPI.

## 9. References
1. Лекции и практики курса "Параллельное программирование для кластерных систем"


## Appendix
ops_seq.cpp:
bool KichanovaKCountLettersInStrSEQ::RunImpl() {
const std::string& input_str = GetInput();
  
  for (char c : input_str) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      GetOutput()++;
    }
  }
  
  return GetOutput() >= 0;
}

ops_mpi.cpp:
bool KichanovaKCountLettersInStrMPI::RunImpl() {
    auto input_str = GetInput();
    if (input_str.empty()) {
        return false;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_length = input_str.length();
    int chunk_size = total_length / size;
    
    int start_index = rank * chunk_size;
    int end_index = (rank == size - 1) ? total_length : start_index + chunk_size;

    int local_count = 0;
    for (int i = start_index; i < end_index; i++) {
        if (std::isalpha(static_cast<unsigned char>(input_str[i]))) {
            local_count++;
        }
    }

    int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GetOutput() = global_count;

    return true;
}
``` 