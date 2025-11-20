# Подсчет частоты символа в строке

- Student: Маслова Ульяна Александровна, group 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 23

## 1. Introduction
Проблема: Последовательный подсчет частоты символов в строках большого размера является медленным.
Задача: Ускорить этот процесс с помощью параллельных вычислений на MPI.
Ожидаемый результат: Значительное сокращение времени выполнения по сравнению с последовательной версией.

## 2. Problem Statement
Нужно найти число вхождений символа input_char в строке input_str.
- InPut: Пара (std::string, char).
- OutPut: Целое число (size_t).

## 3. Baseline Algorithm (Sequential)
Проход по строке в цикле с увеличением счетчика при нахождении искомого символа. Алгоритм имеет линейную временную сложность O(N), где N — длина строки.

## 4. Parallelization Scheme
Процесс с рангом 0 делит исходную строку на P (число процессов) частей. Далее ранг 0 рассылает каждому процессу его фрагмент строки. Каждый процесс независимо считает символы в своей части. Локальные счетчики суммируются на ранге 0.

## 5. Implementation Details
- common: Определяет общие типы данных (InType, OutType).
- seq: Содержит простую последовательную реализацию алгоритма.
- mpi: Содержит параллельную MPI-реализацию алгоритма.
- tests: Включает два набора тестов: functional для проверки корректности и performance для замера скорости.

## 6. Experimental Setup
- Аппаратное обеспечение: AMD Ryzen 7 7840HS (8 ядер, 16 логических процессоров, базовая частота 3,80 ГГц)
- ОЗУ — 16 ГБ 
- Операционная система: Windows 11
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность проверялась на строках различной длинны, а также различного типа (пустые, из одного слова, только из букв, смешанные и т.д.)

### 7.2 Performance

Тест на данных, состоящих из 20 000 000 символов:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.01065 | 1.00    | N/A        |
| mpi         | 2     | 0.01625 | 0.66    | 33.0%      |
| mpi         | 4     | 0.01560 | 0.68    | 17.0%      |
| mpi         | 8     | 0.00848 | 1.26    | 15.8%      |

## 8. Conclusions
- На 2 и 4 процессах: Параллельная программа работает медленнее, чем последовательная. Отправка больших блоков данных между процессами занимает больше времени, чем выигрыш от параллельных вычислений.
- На 8 процессах: Программа начинает работать быстрее последовательной. Задача дробится на достаточно мелкие части, и выигрыш в скорости наконец-то перевешивает затраты на коммуникацию.

## 9. References
1. Лекции и практики курса "Параллельное программирование"

## Appendix (Optional)
```cpp
bool MaslovaUCharFrequencyCountMPI::RunImpl() {
  int rank = 0;
  int proc_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // id процесса
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size); // количество процессов

  std::string input_string;
  char input_char = 0;
  size_t input_str_size = 0;

  if (rank == 0) { 
    input_string = GetInput().first;
    input_char = GetInput().second;
    input_str_size = input_string.size(); // получили данные
    
    if (input_string.empty()) {
      GetOutput() = 0; //если строка пустая, выводим сразу 0
    }
  }

  MPI_Bcast(&input_str_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD); // отправляем размер строки
  if (input_str_size == 0) {
    return true; 
  }

  MPI_Bcast(&input_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD); // отправляем нужный символ
  
  std::vector<int> send_counts(proc_size); //здесь размеры всех порций
  std::vector<int> displs(proc_size); //смещения
  if (rank == 0) {
    size_t part = input_str_size / proc_size;
    size_t rem = input_str_size % proc_size;
    for (int i = 0; i < proc_size; ++i) {
      send_counts[i] = part + (i < rem ? 1 : 0); //общий размер, включающий остаток, если он входит
    }
    displs[0] = 0;
    for (int i = 1; i < proc_size; ++i) {
      displs[i] = displs[i-1] + send_counts[i-1];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size, MPI_INT, 0, MPI_COMM_WORLD); //отправляем размеры порций
  std::vector<char> local_str(send_counts[rank]);
  MPI_Scatterv(
      (rank == 0) ? input_string.data() : nullptr,
      send_counts.data(), displs.data(), MPI_CHAR,
      local_str.data(), local_str.size(), MPI_CHAR,
      0, MPI_COMM_WORLD //распределяем данные
  );

  size_t local_count = std::count(local_str.begin(), local_str.end(), input_char);

  size_t global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); //собрали данные со всех процессов
  
  GetOutput() = global_count; //вывели результат

  return true;
}
```