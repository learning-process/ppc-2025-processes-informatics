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
Для минимизации взаимодействий выбрана одномерная вертикальная декомпозиция (разделение на полосы по столбцам) с обменом halo-зонами.

1. **Распределение нагрузки:** Изображение делится на $N$ вертикальных полос. Каждый процесс вычисляет свою часть столбцов.
2. **Обработка границ:** Каждый процесс добавляет halo-зоны (+1 столбец с каждой стороны) для корректного применения фильтра 3x3.
3. **Коммуникации:**
   - `MPI_Bcast` для передачи метаданных (размеры изображения)
   - `MPI_Scatterv` для распределения данных по процессам
   - `MPI_Sendrecv` для обмена halo-зонами между соседними процессами
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
| SEQ        |    1     |   0.12715   |   1.00×   |     100%      |
| MPI        |    2     |   0.08160   |   1.56×   |     78%       |
| MPI        |    3     |   0.06466   |   1.97×   |     66%       |
| MPI        |    4     |   0.06033   |   2.11×   |     53%       |

### 5.3 Анализ результатов

1. **Линейное масштабирование:**
   - На 2 процессах: ускорение 1.56× (сокращение времени на 36%)
   - На 3 процессах: ускорение 1.97× (почти двукратное ускорение)
   - На 4 процессах: ускорение 2.11× (максимальное ускорение)

2. **Эффективность параллелизации:**
   - 2 процесса: 78% - высокая эффективность
   - 3 процесса: 66% - хорошая эффективность  
   - 4 процесса: 53% - умеренная эффективность

3. **Особенности реализации:**
   - Вертикальное разбиение с обменом halo-зонами
   - Корректная обработка границ между процессами
   - Минимальные коммуникационные затраты

4. **Оптимальность конфигурации:**
   - Наилучшее соотношение производительности достигается на 3 процессах
   - Дальнейшее увеличение процессов даёт меньший прирост производительности
   - Алгоритм демонстрирует хорошее масштабирование до 4 процессов

## 6. Заключение
Разработана параллельная реализация фильтра Гаусса для обработки изображений с использованием MPI. Алгоритм реализует вертикальное разбиение изображения с обменом граничными данными между процессами.

**Основные результаты:**
- Достигнуто ускорение до 2.11× на 4 процессах
- Наилучшая эффективность 78% при 2 процессах
- Корректная обработка границ изображения через обмен halo-зонами

**Преимущества реализации:**
- Хорошее масштабирование до 4 процессов
- Минимальные коммуникационные накладные расходы
- Устойчивая работа при различном количестве процессов

Разработанное решение эффективно для обработки больших изображений на системах с несколькими вычислительными ядрами.

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
    
    std::vector<int> proc_cols_count(size);
    std::vector<int> proc_start_col(size);
    
    if (rank == 0) {
        int current_col = 0;
        for (int proc = 0; proc < size; ++proc) {
            const bool gets_extra = proc < remainder;
            proc_cols_count[proc] = gets_extra ? cols_per_proc + 1 : cols_per_proc;
            proc_start_col[proc] = current_col;
            current_col += proc_cols_count[proc];
        }
    }
    
    MPI_Bcast(proc_cols_count.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(proc_start_col.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    const int local_cols = proc_cols_count[rank];
    
    const int halo_size = 1;
    const int local_cols_with_halo = local_cols + (2 * halo_size);
    const int local_size = local_cols * rows * channels;
    
    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);
    
    if (rank == 0) {
        int offset = 0;
        for (int proc = 0; proc < size; ++proc) {
            send_counts[proc] = proc_cols_count[proc] * rows * channels;
            displacements[proc] = offset;
            offset += send_counts[proc];
        }
    }
    
    std::vector<uint8_t> local_data(local_size);
    std::vector<uint8_t> local_out(local_size);
    
    const uint8_t* send_buffer = rank == 0 ? GetInput().data.data() : nullptr;
    MPI_Scatterv(send_buffer, send_counts.data(), displacements.data(), MPI_BYTE,
                 local_data.data(), local_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto local_data_with_halo = PrepareLocalDataWithHalo(local_data, rows, local_cols,
                                                         local_cols_with_halo, channels,
                                                         halo_size);
    
    ExchangeHaloZones(local_data_with_halo, rows, local_cols, local_cols_with_halo,
                      channels, halo_size, rank, size);
    
    ComputeLocalGaussWithHalo(local_data_with_halo, local_out, rows, local_cols, 
                              local_cols_with_halo, channels, halo_size);
    
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
