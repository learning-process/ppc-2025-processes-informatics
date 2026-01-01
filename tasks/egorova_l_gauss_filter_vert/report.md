# Вертикальный фильтр Гаусса (3x3)
- **Студент:** Егорова Лариса Алексеевна
- **Группа:** 3823Б1ФИ1
- **Технология:** MPI, SEQ
- **Вариант:** 27

## 1. Введение
Фильтрация изображений является одной из ключевых задач компьютерного зрения. Фильтр Гаусса применяется для снижения уровня цветового шума и сглаживания деталей. В условиях работы с изображениями высокого разрешения последовательная обработка становится узким местом, что обуславливает необходимость использования технологий параллельного программирования, таких как MPI.

## 2. Постановка задачи
Необходимо реализовать параллельный алгоритм фильтрации изображения с использованием ядра Гаусса размером $3 \times 3$.
- **Вход:** Структура данных со строками, столбцами, каналами и вектором пикселей `uint8_t`.
- **Выход:** Преобразованный вектор пикселей того же размера.
- **Особенности:** Реализация должна корректно обрабатывать границы изображения (стратегия `clamp`) и обеспечивать синхронизацию "призрачных зон" (halo-зоны) между процессами.

## 3. Последовательный алгоритм
Алгоритм реализует математическую операцию свертки изображения с дискретным аналогом функции Гаусса.
1. Для каждого пикселя $(x, y)$ анализируется окрестность $3 \times 3$.
2. Используется ядро с весовыми коэффициентами и последующей нормализацией ($\frac{1}{16}$):
   $$K = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
3. На краях изображения используется значение ближайшего существующего пикселя.

## 4. Схема распараллеливания (MPI)
Для минимизации взаимодействий выбрана одномерная вертикальная декомпозиция (разделение на полосы по столбцам).

1. **Распределение нагрузки:** Изображение делится на $N$ вертикальных полос. Каждый процесс вычисляет свою часть столбцов.
2. **Halo-зоны:** Каждому процессу требуется один соседний столбец от предыдущего и последующего процессов для корректного расчета границ своей полосы.
3. **Коммуникации:**
   - `MPI_Bcast` для передачи метаданных.
   - `MPI_Scatterv` для раздачи полос изображения процессам.
   - `MPI_Sendrecv` для обмена граничными столбцами.
   - `MPI_Gatherv` для финальной сборки результата на мастере.

## 5. Эксперименты

### 5.1 Среда проведения тестов

- **Процессор:** 13th Gen Intel(R) Core(TM) i5-13420H (2.10 GHz, 8 ядер / 12 потоков)
- **Оперативная память:** 16,0 ГБ
- **ОС:** Windows 11 (WSL2 / Docker Desktop)
- **Инструментарий:** Clang-21, MPI, CMake, Release build.

### 5.2 Производительность

Замеры проводились в режиме `task_run`. Ускорение ($S$) рассчитано относительно последовательной реализации (SEQ), где $T_{SEQ} = 0.03340$ сек.

$$S = \frac{T_{SEQ}}{T_P} \quad E = \frac{S}{P} \times 100\%$$

**Таблица 1. Результаты измерений производительности**

| Реализация | Процессы ($P$) | Время $T$ (сек) | Ускорение $S$ | Эффективность $E$ |
| :--------- | :------------: | :-------------: | :-----------: | :---------------: |
| **SEQ**    |       1        |     0.03340     |     1.00      |       100%        |
| **MPI**    |       2        |     0.02341     |     1.43      |       71%         |
| **MPI**    |       3        |     0.01699     |     1.97      |       66%         |
| **MPI**    |       4        |     0.01979     |     1.69      |       42%         |

### 5.3 Анализ результатов

* **Максимальное ускорение** (**1.97×**) достигнуто на **3 процессах**. Параллельная реализация стабильно работает быстрее последовательной на конфигурациях с 2 и 3 ядрами.
* **Эффективность** снижается с ростом числа процессов (с 71% до 42%). Это указывает на то, что, хотя вычисления параллелизуются, накладные расходы на коммуникацию (обмен **halo-зонами** и коллективные операции `MPI_Scatterv`/`MPI_Gatherv`) становятся существенными при дальнейшем дроблении задачи.
* **Масштабируемость:** На 4 процессах наблюдается **падение производительности** по сравнению с 3 процессами (0.01979с против 0.01699с). Это ключевой признак того, что на данном объеме данных коммуникационные издержки начинают доминировать над выгодой от параллелизма.

## 6. Заключение

Разработанный параллельный алгоритм вертикального фильтра Гаусса с использованием MPI подтвердил свою работоспособность и эффективность.

Оптимальное количество процессов для предоставленных тестовых данных — 3. При этом достигается наилучший баланс между временем вычислений и накладными расходами на коммуникацию, обеспечивая ускорение, близкое к линейному (1.97×). Реализованный механизм обмена `halo`-зонами обеспечивает корректную обработку границ между данными процессов.

## 7. Источники
1. Open MPI Documentation: [https://www.open-mpi.org/doc/](https://www.open-mpi.org/doc/)
2. MPI Standard: [https://www.mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
3. Introduction to MPI: [https://mpitutorial.com/](https://mpitutorial.com/)
4. Материалы курса «Параллельное программирование».

## Приложение (код)

```cpp

bool EgorovaLGaussFilterVertMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rr = GetInput().rows;
  int cc = GetInput().cols;
  int ch = GetInput().channels;

  MPI_Bcast(&rr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ch, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int qq = cc / size;
  int rem = cc % size;
  int local_cols = qq + (rank < rem ? 1 : 0);

  std::vector<int> counts(static_cast<size_t>(size));
  std::vector<int> displs(static_cast<size_t>(size));
  CalculateCountsAndDispls(size, qq, rem, rr, ch, counts, displs);

  std::vector<uint8_t> local_data(static_cast<size_t>(rr) * local_cols * ch);

  MPI_Scatterv(rank == 0 ? GetInput().data.data() : nullptr, counts.data(), displs.data(), MPI_BYTE, local_data.data(),
               static_cast<int>(local_data.size()), MPI_BYTE, 0, MPI_COMM_WORLD);

  int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
  int left_h = (left_rank != MPI_PROC_NULL) ? 1 : 0;
  int right_h = (right_rank != MPI_PROC_NULL) ? 1 : 0;
  int total_lc = local_cols + left_h + right_h;

  std::vector<uint8_t> local_with_halo(static_cast<size_t>(rr) * total_lc * ch);
  FillLocalWithHalo(local_data, local_with_halo, rr, local_cols, total_lc, left_h, ch);

  ExchangeHalo(local_with_halo, right_rank, rr, total_lc, ch, static_cast<size_t>(left_h + local_cols - 1),
               static_cast<size_t>(left_h + local_cols));
  ExchangeHalo(local_with_halo, left_rank, rr, total_lc, ch, static_cast<size_t>(left_h), 0);

  std::vector<uint8_t> local_out(static_cast<size_t>(rr) * local_cols * ch);
  ApplyFilter(local_with_halo, local_out, rr, local_cols, total_lc, ch);

  if (rank == 0) {
    GetOutput().rows = rr;
    GetOutput().cols = cc;
    GetOutput().channels = ch;
    GetOutput().data.resize(static_cast<size_t>(rr) * cc * ch);
  }

  MPI_Gatherv(local_out.data(), static_cast<int>(local_out.size()), MPI_BYTE,
              rank == 0 ? GetOutput().data.data() : nullptr, counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  return true;
}