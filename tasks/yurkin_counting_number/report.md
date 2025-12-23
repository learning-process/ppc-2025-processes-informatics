# Отчёт

# Подсчёт количества букв во входных данных. Последовательная и MPI реализации

- **Студент:** Юркин Георгий Алексеевич, группа 3823Б1ФИ1  
- **Технология:** SEQ–MPI  
- **Вариант:** 22  

---

## 1. Введение

В данной работе рассматривается задача подсчёта количества букв (A–Z, a–z) во входных данных. Задача является типовой для анализа массивов данных и хорошо подходит для демонстрации возможностей последовательных и параллельных вычислений.

Целью работы является реализация последовательной (SEQ) и параллельной (MPI) версий алгоритма подсчёта, проверка корректности их работы и сравнение производительности.
---

## 2. Постановка задачи

Необходимо выполнить подсчёт количества букв во входных данных.

### Входные данные
- `input`— массив символов (std::vector<char>) 

### Выходные данные
- `result` — количество букв в массиве  

### Требования
- Реализация SEQ и MPI версий  
- Корректная работа при любом числе процессов  
- Совпадение результатов SEQ и MPI реализаций  

---

## 3. Последовательный алгоритм

Последовательная версия алгоритма выполняет вычисления в одном потоке:

1. Получение входных данных  
2. Инициализация счётчика  
3. Последовательный перебор входных данных  
4. Увеличение счётчика при выполнении условия  
5. Возврат итогового значения  

Алгоритм имеет линейную вычислительную сложность и не требует дополнительной памяти.

---

## 4. Параллельный алгоритм (MPI)

### Идея распараллеливания

- Входные данные логически делятся между MPI-процессами  
- Каждый процесс обрабатывает свою часть данных  
- Вычисляются локальные частичные результаты  
- Глобальный результат формируется с помощью `MPI_Reduce`  

### Синхронизация
- Используется `MPI_COMM_WORLD`  
- Объединение частичных результатов выполняется коллективной операцией MPI_Reduce

---
### Псевдокод

```
MPI_Comm_rank(MPI_COMM_WORLD, &rank)
MPI_Comm_size(MPI_COMM_WORLD, &size)

if rank == 0:
    N = GetInput().size()
MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD)

compute sendcounts and displs based on N and size

local_buf = vector<char>(sendcounts[rank])
MPI_Scatterv(root_buffer, sendcounts, displs, MPI_CHAR,
             local_buf, sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD)

local_count = 0
for c in local_buf:
    if isalpha(static_cast<unsigned char>(c)):
        local_count += 1

global_count = 0
MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD)

GetOutput() = global_count



```
---

## 5. Детали реализации

### Структура проекта
- `ops_seq.cpp` — последовательная реализация  
- `ops_mpi.cpp` — параллельная MPI-реализация  
- `common.hpp` — описание типов входных и выходных данных  
- `tests/functional` — функциональные тесты  
- `tests/performance` — тесты производительности  

### Особенности
- MPI-реализация использует Scatterv для распределения входного массива между процессами
- После локального подсчёта букв используется Allreduce для получения глобального результата 
- Логика SEQ и MPI версий максимально совпадает  

---

## 6. Экспериментальная установка

### Аппаратное обеспечение
- CPU: AMD RYZEN 5500U
- RAM: 16 ГБ  
- OS: Windows 10   
- MPI: OpenMPI / MS-MPI  

### Параметры тестирования
- Различные размеры входных данных  
- Количество процессов: 1, 2, 4  

---

## 7. Результаты

### 7.1 Корректность

- Результаты SEQ и MPI совпадают  
- Функциональные тесты пройдены  
- Ошибки синхронизации отсутствуют  

### 7.2 Производительность

Для оценки производительности были проведены замеры времени выполнения последовательной и MPI-версий алгоритма подсчёта. Тестирование выполнялось на больших объёмах входных данных, сгенерированных автоматически в тестах производительности.

| Mode | Processes | Размер входных данных | Time (s) | Speedup |
|------|-----------|-----------------------|----------|---------|
| seq  | 1         | 1 × 10⁶               | 0.095    | 1.00    |
| mpi  | 1         | 1 × 10⁶               | 0.102    | 0.93    |
| mpi  | 2         | 1 × 10⁶               | 0.056    | 1.70    |
| mpi  | 4         | 1 × 10⁶               | 0.031    | 3.06    |

**Обсуждение:**  
MPI-версия показывает ускорение при использовании нескольких процессов за счёт распараллеливания подсчёта. При одном процессе MPI уступает последовательной версии из-за накладных расходов на инициализацию и коммуникации. С ростом числа процессов и размера входных данных эффективность параллелизации возрастает, однако масштабируемость ограничена затратами на коллективные операции (`MPI_Reduce`).


## 8. Заключение

В ходе работы была реализована задача подсчёта количества элементов в последовательной и параллельной (MPI) версиях.

- Подтверждена корректность работы  
- Показана применимость MPI для задач подсчёта  
- Получено ускорение на больших входных данных  

---

## 9. References

1. MPI Standard — https://www.mpi-forum.org  
2. cppreference.com — https://en.cppreference.com  
3. Документация OpenMPI / MS-MPI  


## Appendix (Optional)

### MPI-код

```
int world_rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

int total_size = 0;
if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
}
MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

std::vector<char> buffer;
if (world_rank == 0 && total_size > 0) {
    buffer.assign(GetInput().begin(), GetInput().end());
}

std::vector<int> sendcounts(static_cast<std::size_t>(world_size), 0);
std::vector<int> displs(static_cast<std::size_t>(world_size), 0);
if (world_size > 0 && total_size > 0) {
    int base = total_size / world_size;
    int rem = total_size % world_size;
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) {
        displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
}

int local_count_chars = sendcounts[world_rank];
std::vector<char> local_buf(static_cast<std::size_t>(local_count_chars));

MPI_Scatterv(
    (buffer.empty() ? nullptr : buffer.data()),
    sendcounts.data(),
    displs.data(),
    MPI_CHAR,
    (local_buf.empty() ? nullptr : local_buf.data()),
    local_count_chars,
    MPI_CHAR,
    0,
    MPI_COMM_WORLD);

int local_count = 0;
for (char c : local_buf) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
        ++local_count;
    }
}

int global_count = 0;
MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

GetOutput() = global_count;



```