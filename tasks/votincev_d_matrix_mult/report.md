# Отчет по реализации алгоритма ленточной горизонтальной схема - разбиение только матрицы А - умножение матрицы на матрицу

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Вотинцев Дмитрий Сергеевич 3823Б1ФИ3
**Вариант:** 13

## Введение
В рамках данной работы был реализован алгоритм умножения матриц с использованием ленточной горизонтальной схемы, где матрица A делится по строкам, а матрица B остается без изменений.


## Постановка задачи
Задача заключается в умножении матриц A и B, используя ленточно-горизонтальную схему - разбиение только матрицы А. Это значит, что матрица A делится по строкам и производится умножение каждой строки на матрицу B.



## Описание алгоритма
SEQ-версия: обычное умножение матрицы A на матрицу B.



## Описание схемы параллельного алгоритма
MPI-версия: 
- Делим матрицу A по строкам
- Распределяем строки между процессами
- Каждый процесс умножает свои строки на матрицу B и записывает результат в свою локальную матрицу
- 0й процесс собирает результирующую матрицу


## Результаты экспериментов и выводы

Тест1: нулевые матрицы
Тест2: 1x9x9
Тест3: 9x1x9
Тест4: 9x9x1
Тест5: произведение квадратных (4x4x4)
Тест6: произведение квадратных побольше (6x6x6)
Тест7-10: произведение прямоугольных матриц

Матрицы: A (m x k) B (k x n) R (m x n) 
m - кол-во строк в матрице A и R
n - кол-во столбцов в матрце B и R
k - кол-во столбцов в матрице A, строк в матрице B

| MxNxK         | SEQ версия (с) | MPI версия (с) | Ускорение |
|---------------|----------------|----------------|-----------|
| 100           | 0.058          | 0.032          | 1.81×     |
| 200           | 0.409          | 0.172          | 2.38×     |
| 300           | 1.448          | 0.558          | 2.60×     |
| 400           | 3.542          | 1.373          | 2.58×     |
| 500           | 7.885          | 3.051          | 2.58×     |
| 600           | 13.271         | 5.674          | 2.34×     |

MPI версия показывает в среднем ускорение 2.5x в сравнении с MPI. Эта задачка - не как первая, где особо вычислений нет и MPI намного хуже из-за пересылок данных (упор в память).
Здесь есть вычислительная часть - умножение матриц имеет сложность O(n^3) (если m,n,k разные, то O(MxNxK)).

- M=N=K=100: 1 000 000 операций
- M=N=K=200: 8 000 000 операций
- M=N=K=600: 216 000 000 операций
Последовательная версия в случае с M=N=K=600 выполняет все 216млн операций одна. Если же мы задействуем 4 потока и не учитываем накладные расходы от пересылок MPI, то в среднем поток обрабатывает чистые 54 млн операций и тогда ускорение 4x.
Но MPI тоже занимает время, пересылки строк матриц занимают время, и, как видно, существенно (хотя ускорение в 2.5x это даже очень хорошо).

## Заключение
В результате проделанной работы была реализована ленточно-горизонтальная схема умножения матриц (где A разбивается на строки, B остается без изменений).
На основе проведенных экспериментов можно сказать точно: MPI лучше версии SEQ и показывает стабильное ускорение в 2.5x раза.





## Литература
1. Лекции Сысоева Александра Владимировича
2. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
3. Интернет
4. Умножение, хранение матрицы (https://hpc-education.unn.ru/files/courses/optimization/1_4_Compiler.pdf)


## Приложения (код параллельной реализации)
```
bool VotincevDMatrixMultMPI::RunImpl() {
  // получаю кол-во процессов
  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);

  // получаю ранг текущего
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // размерности матриц получают все процессы
  int m = 0, n = 0, k = 0;
  const auto& in = GetInput();
  m = std::get<0>(in);
  n = std::get<1>(in);
  k = std::get<2>(in);

  // если процессов больше чем строк в матрице A - 
  // то активных процессов будет меньше (m)
  //  (потому что разедление по строкам)
  process_n = std::min(process_n, m);
  
  // "лишние" процессы не работают
  if(proc_rank >= process_n) {
    return true;
  }
  
  std::vector<double> matrix_A;
  std::vector<double> matrix_B;

  // матрицу B получают все процессы
  matrix_B = std::get<4>(in);

  // матрицу А получит полностью только 0й процесс
  if (proc_rank == 0) {
    matrix_A = std::get<3>(in);
  }

  // если всего 1 процесс - последовательное умножение
  if (process_n == 1) {
    GetOutput() = SeqMatrixMult(m,n,k,matrix_A, matrix_B);
    return true;
  }

  // какие строки каждый процесс будет перемножать
  // [start0, end0, start1, end1, ...]
  std::vector<int> ranges;

  if (proc_rank == 0) {
    ranges.resize(process_n * 2);

    // минимум на обработку
    int base = m / process_n;
    // остаток распределим
    int remain = m % process_n;

    int start = 0;
    for (int i = 0; i < process_n; i++) {
      int part = base;
      if (i < remain) {
        part++;
      }

      ranges[i * 2] = start;
      ranges[i * 2 + 1] = start + part;  // end (не включительно)

      start += part;
    }
  }

  // my_range получит [start, end]
  int my_range[2]{0, 0};

  // local_matrix — локальный блок матрицы A данного процесса
  std::vector<double> local_matrix;

  if (proc_rank == 0) {
    // заполняю данные для себя — свои строки матрицы А
    int my_start = ranges[0];
    int my_end   = ranges[1];
    int my_rows  = my_end - my_start;

    local_matrix.resize(my_rows * k);
    for (int i = 0; i < my_rows * k; i++) {
      local_matrix[i] = matrix_A[i];
    }


    // рассылаю остальным
    for (int i = 1; i < process_n; i++) {
      int start_i = ranges[2 * i];
      int end_i   = ranges[2 * i + 1];

      MPI_Send(&start_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&end_i,   1, MPI_INT, i, 0, MPI_COMM_WORLD);

      int elem_count = (end_i - start_i) * k;

      MPI_Send(matrix_A.data() + start_i * k,
               elem_count,
               MPI_DOUBLE,
               i,
               0,
               MPI_COMM_WORLD);
    }

  } else {
    // получаю диапазон
    MPI_Recv(&my_range[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&my_range[1], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_i = my_range[0];
    int end_i   = my_range[1];
    int elem_count = (end_i - start_i) * k;

    local_matrix.resize(elem_count);

    // получаю матрицу (часть)
    MPI_Recv(local_matrix.data(),
             elem_count,
             MPI_DOUBLE,
             0,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  
  // теперь каждый владеет своим куском (local_matrix)
  // вызываем обычное перемножение, результат умножение кладется в local_matrix
  MatrixPartMult(k, n, local_matrix, matrix_B); 

  // сбор результатов назад к 0му
    if (proc_rank == 0) {
        std::vector<double> final_result(m * n);
        int my_rows = ranges[1] - ranges[0]; // сколько строк
        // копирую        откуда          до куда                 куда
        std::copy(local_matrix.begin(), local_matrix.end(), final_result.begin());

        // смещение; изначально равно количеству уже записанных значений
        int offset = my_rows * n; 
        for (int i = 1; i < process_n; ++i) {
            int start_i = ranges[2 * i];
            int end_i   = ranges[2 * i + 1];
            int rows    = end_i - start_i;

            // сколько в данной пачке элементов
            int count   = rows * n; 

            MPI_Recv(final_result.data() + offset,
                     count,
                     MPI_DOUBLE,
                     i,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            offset += count;
        }
        // 0й процесс собрал всё от других процессов

        GetOutput() = final_result;
        
    } else {
      // другие процессы посылают свои результаты основному 0му процессу
      int rows = my_range[1] - my_range[0];
      MPI_Send(local_matrix.data(), rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }


  return true;
}
```