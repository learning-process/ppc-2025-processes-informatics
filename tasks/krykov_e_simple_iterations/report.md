# Отчёт по лабораторной работе  
## «Решение СЛАУ методом простых итераций с использованием MPI»  

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Крюков Егор Федорович 3823Б1ФИ1
**Вариант:** 20


## Введение

Решение систем линейных алгебраических уравнений (СЛАУ) является одной из фундаментальных задач вычислительной математики. Метод простых итераций — это итерационный численный метод, который позволяет находить приближённое решение СЛАУ. Параллельные вычисления позволяют ускорить процесс решения больших систем уравнений, что актуально для задач моделирования, обработки данных и машинного обучения.

В рамках данной работы были реализованы последовательная и параллельная версии метода простых итераций для решения СЛАУ.


## Постановка задачи

Задача заключается в решении системы линейных уравнений вида: Ax = b, где A — квадратная матрица коэффициентов размера n, b — вектор правых частей, x — искомый вектор решения.

Метод простых итераций использует итерационную формулу. Итерации продолжаются до достижения заданной точности Eps: x(k+1) - x(k)  < Eps


## Описание алгоритма

Базовый последовательный алгоритм начинается с инициализации начального приближения вектора x нулями. На каждой итерации для каждого i вычисляется новое значение x_i по итерационной формуле. После завершения итераций для всех компонент вычисляется норма разности между новым и старым вектором решения. Если норма меньше заданной точности Eps, процесс останавливается. В противном случае итерации продолжаются до достижения максимального числа итераций.

## Описание схемы параллельного алгоритма

Параллельная реализация с использованием MPI основывается на распределении строк матрицы A и соответствующих компонент векторов b и x между процессами. Каждый процесс вычисляет новые значения для своей части вектора x. После этого с помощью операций MPI (MPI_Allgatherv) обновлённые значения собираются со всех процессов, чтобы каждый процесс имел полный актуальный вектор x для следующей итерации. Норма разности вычисляется локально на каждом процессе для его части, а затем находится глобальная норма. Итерации продолжаются до достижения заданной точности.


## Результаты экспериментов

Условия экспериментов:  
Размеры тестовой матрицы n: 20, 200, 500.  
Среда выполнения: Windows, MPI (4 процесса).
Измерение времени — средствами тестового окружения GoogleTest (режимы pipeline и task_run).


### Результаты при  n = 20

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00029500 | Параллельная версия     |
| MPI task_run     |  0.00018992 | Параллельная версия     |
| SEQ pipeline     |  0.00001736 | Последовательная версия |
| SEQ task_run     |  0.00001700 | Последовательная версия |


Вывод: При малых размерах матрицы последовательная версия работает быстрее, так как накладные расходы на коммуникацию в MPI превышают выгоду от параллелизма. 


### Результаты при \( n = 200 \)

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00725706 | Параллельная версия     |
| MPI task_run     |  0.00717490 | Параллельная версия     |
| SEQ pipeline     |  0.01925320 | Последовательная версия |
| SEQ task_run     |  0.01900000 | Последовательная версия |

Вывод: при увеличении размера системы параллельная версия становится быстрее примерно в 2.7 раза. Накладные расходы MPI становятся менее значительными.

### Результаты при \( n = 500 \)

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.09830734 | Параллельная версия     |
| MPI task_run     |  0.09841740 | Параллельная версия     |
| SEQ pipeline     |  0.33315890 | Последовательная версия |
| SEQ task_run     |  0.33300000 | Последовательная версия |

Вывод: на больших размерах системы ускорение MPI достигает примерно 3.4 раза, что демонстрирует эффективность параллельного алгоритма.


## Заключение

В ходе работы успешно реализованы последовательный и параллельный алгоритмы решения СЛАУ методом простых итераций. Параллельная реализация с использованием MPI показывает значительное ускорение на больших размерах системы, однако на малых размерах накладные расходы на коммуникацию делают последовательную версию более эффективной. Оптимальная область применения параллельного алгоритма — задачи с большими системами уравнений.


## Источники

1. Лекции Сысоева Александра Владимировича
2. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
3. Документация по курсу
4. Различные источники из интернета


## Приложение. Исходный код

### SEQ-версия

```cpp
namespace krykov_e_simple_iterations {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

KrykovESimpleIterationsSEQ::KrykovESimpleIterationsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsSEQ::ValidationImpl() {
  const auto &[n, A, b] = GetInput();
  return n > 0 && A.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsSEQ::RunImpl() {
  const auto &[n, A, b] = GetInput();

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[(i * n) + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[(i * n) + i];
    }

    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = x_new[i] - x[i];
      norm += diff * diff;
    }

    x = x_new;

    if (std::sqrt(norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationsSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace krykov_e_simple_iterations

```

### MPI-версия
```cpp
namespace krykov_e_simple_iterations {
namespace {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

void CalculateLocalXNew(int start, int count, size_t n, const std::vector<double> &local_a,
                        const std::vector<double> &local_b, const std::vector<double> &x,
                        std::vector<double> &local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (std::cmp_not_equal(j, global_i)) {
        sum += local_a[i * n + j] * x[j];
      }
    }
    local_x_new[i] = (local_b[i] - sum) / local_a[i * n + global_i];
  }
}

double CalculateLocalNorm(int start, int count, const std::vector<double> &x_new, const std::vector<double> &x) {
  double local_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    int gi = start + i;
    double diff = x_new[gi] - x[gi];
    local_norm += diff * diff;
  }
  return local_norm;
}

void CalculateChunkSizesAndDispls(int size, int n, std::vector<int> &chunk_sizes, std::vector<int> &displs) {
  int base = n / size;
  int rem = n % size;

  displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    chunk_sizes[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + chunk_sizes[i - 1];
    }
  }
}

void CalculateMatrixChunkSizesAndDispls(int size, int n, std::vector<int> &matrix_chunk_sizes,
                                        std::vector<int> &matrix_displs) {
  int base = n / size;
  int rem = n % size;

  matrix_displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    int rows = base + (i < rem ? 1 : 0);
    matrix_chunk_sizes[i] = rows * n;
    if (i > 0) {
      matrix_displs[i] = matrix_displs[i - 1] + matrix_chunk_sizes[i - 1];
    }
  }
}

}  // namespace

KrykovESimpleIterationsMPI::KrykovESimpleIterationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsMPI::ValidationImpl() {
  const auto &[n, a, b] = GetInput();
  return n > 0 && a.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t n = 0;
  std::vector<double> a;
  std::vector<double> b;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a = std::get<1>(input);
    b = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_sizes(size);
  std::vector<int> displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), chunk_sizes, displs);

  std::vector<int> matrix_chunk_sizes(size);
  std::vector<int> matrix_displs(size);
  CalculateMatrixChunkSizesAndDispls(size, static_cast<int>(n), matrix_chunk_sizes, matrix_displs);

  int local_rows = chunk_sizes[rank];
  int local_matrix_size = matrix_chunk_sizes[rank];

  std::vector<double> local_a(local_matrix_size);
  std::vector<double> local_b(local_rows);

  MPI_Scatterv(rank == 0 ? a.data() : nullptr, matrix_chunk_sizes.data(), matrix_displs.data(), MPI_DOUBLE,
               local_a.data(), local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_DOUBLE, local_b.data(),
               local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start = displs[rank];
  int count = local_rows;

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(size);
  std::vector<int> allgather_displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), recv_counts, allgather_displs);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, local_a, local_b, x, local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), allgather_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_norm = CalculateLocalNorm(start, count, x_new, x);
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;

    if (std::sqrt(global_norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool KrykovESimpleIterationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_simple_iterations


```