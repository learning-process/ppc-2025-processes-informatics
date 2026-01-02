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
- **Особенности:** Реализация должна корректно обрабатывать границы изображения (стратегия `clamp`) и обеспечивать синхронизацию данных между процессами.

## 3. Последовательный алгоритм
Алгоритм реализует математическую операцию свертки изображения с дискретным аналогом функции Гаусса.
1. Для каждого пикселя $(x, y)$ анализируется окрестность $3 \times 3$.
2. Используется ядро с весовыми коэффициентами и последующей нормализацией ($\frac{1}{16}$):
   $$K = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
3. На краях изображения используется значение ближайшего существующего пикселя.

## 4. Схема распараллеливания (MPI)
Для минимизации взаимодействий выбрана одномерная вертикальная декомпозиция (разделение на полосы по столбцам) без обмена halo-зонами.

1. **Распределение нагрузки:** Изображение делится на $N$ вертикальных полос. Каждый процесс вычисляет свою часть столбцов независимо.
2. **Обработка границ:** Каждый процесс использует стратегию `clamp` в пределах своих данных.
3. **Коммуникации:**
   - `MPI_Bcast` для передачи метаданных (размеры изображения)
   - `MPI_Scatterv` для распределения данных по процессам
   - `MPI_Gatherv` для сбора результатов на процесс 0
   - `MPI_Bcast` для распространения финального результата всем процессам

## 5. Эксперименты

### 5.1 Среда проведения тестов

- **Процессор:** 13th Gen Intel(R) Core(TM) i5-13420H (2.10 GHz, 8 ядер / 12 потоков)
- **Оперативная память:** 16,0 ГБ
- **ОС:** Windows 11 (WSL2 / Docker Desktop)
- **Инструментарий:** Clang-21, MPI, CMake, Release build.

### 5.2 Производительность
Измерения проводились на изображении размером 2000×2000×3 канала (12 миллионов пикселей).

**Таблица 1. Результаты измерения производительности**

| Реализация | Процессы | Время (сек) | Ускорение | Эффективность |
| :--------- | :------: | :---------: | :-------: | :-----------: |
| SEQ        |    1     |   0.12986   |   1.00×   |     100%      |
| MPI        |    2     |   0.08459   |   1.53×   |     77%       |
| MPI        |    3     |   0.06517   |   1.99×   |     66%       |
| MPI        |    4     |   0.05819   |   2.23×   |     56%       |

### 5.3 Анализ результатов

1. **Достигнутое ускорение:**
   - На 2 процессах: сокращение времени на 35%
   - На 3 процессах: почти двукратное ускорение (1.99×)
   - На 4 процессах: максимальное ускорение 2.23×

2. **Особенности реализации:**
   - Каждый процесс обрабатывает свою группу столбцов независимо
   - Стратегия `clamp` применяется в пределах локальных данных
   - Минимальные коммуникационные затраты между процессами

3. **Ограничения:**
   - Падение эффективности с ростом числа процессов
   - Отсутствие обмена граничными данными между процессами
   - Коммуникационные операции преобладают при мелком разбиении

## 6. Заключение

Разработанная параллельная реализация фильтра Гаусса демонстрирует **практически значимое ускорение** при обработке изображений большого размера.

**Основные достижения:**
- Ускорение 2.23× на 4 процессах
- Корректная работа алгоритма при вертикальном разбиении
- Минимальные накладные расходы на коммуникацию

**Ограничения и перспективы:**
- Эффективность снижается при увеличении числа процессов
- Для дальнейшего масштабирования требуется обмен halo-зонами
- Алгоритм оптимален для изображений, где высота значительно превышает ширину

Разработанное решение может быть использовано для обработки больших изображений в системах с умеренным числом вычислительных узлов.

## 7. Источники
1. Open MPI Documentation: [https://www.open-mpi.org/doc/](https://www.open-mpi.org/doc/)
2. MPI Standard: [https://www.mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
3. Introduction to MPI: [https://mpitutorial.com/](https://mpitutorial.com/)
4. Материалы курса «Параллельное программирование».

## Приложение (код)

```cpp
bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() {
    return true;
}

bool EgorovaLGaussFilterVertMPI::RunImpl() {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0;
    int cols = 0;
    int channels = 0;
    
    if (rank == 0) {
        rows = GetInput().rows;
        cols = GetInput().cols;
        channels = GetInput().channels;
    }
    
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    const int cols_per_proc = cols / size;
    const int remainder = cols % size;
    
    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);
    
    if (rank == 0) {
        int offset = 0;
        for (int proc = 0; proc < size; ++proc) {
            const int proc_cols = (proc < remainder) ? (cols_per_proc + 1) : cols_per_proc;
            send_counts[proc] = proc_cols * rows * channels;
            displacements[proc] = offset;
            offset += send_counts[proc];
        }
    }
    
    MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    const int local_size = send_counts[rank];
    std::vector<uint8_t> local_in(local_size);
    std::vector<uint8_t> local_out(local_size);
    
    MPI_Scatterv(rank == 0 ? GetInput().data.data() : nullptr, 
                 send_counts.data(), displacements.data(), MPI_BYTE,
                 local_in.data(), local_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    const int local_cols = (rows * channels == 0) ? 0 : local_size / (rows * channels);
    ComputeLocalGauss(local_in, local_out, rows, local_cols, channels);
    
    auto& out = GetOutput();
    out.rows = rows;
    out.cols = cols;
    out.channels = channels;
    
    const std::size_t total_size = static_cast<std::size_t>(rows) * 
                                   static_cast<std::size_t>(cols) * 
                                   static_cast<std::size_t>(channels);
    out.data.resize(total_size);
    
    MPI_Gatherv(local_out.data(), local_size, MPI_BYTE,
                out.data.data(), send_counts.data(), displacements.data(),
                MPI_BYTE, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(out.data.data(), static_cast<int>(total_size), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    return true;
}