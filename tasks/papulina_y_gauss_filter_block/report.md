# Линейная фильтрация изображений (блочное разбиение). Ядро Гаусса 3x3.

- Student: Папулина Юлия Андреевна, group 3823Б1ФИ3
- Technology: SEQ | MPI 
- Variant: 28

## 1. Введение

## 2. Постановка задачи

**Формальная постановка:** 
**Входные данные:**

**Выходные данные:**

**Ограничения:**

## 3. Последовательная версия (Baseline)

## 4. Параллельная версия

### 4.1. Разделение данных

### 4.2. Взаимодействие процессов

**Особенности реализации:**

## 5. Детали реализации

### 5.1. Файловая структура проекта
papulina_y_gauss_filter_block/  
├── common/include/common.hpp  
├── seq/include/ops_seq.hpp  
├── seq/src/ops_seq.cpp  
├── mpi/include/ops_mpi.hpp  
├── mpi/src/ops_mpi.cpp  
├── tests/functional/main.cpp  
├── tests/performance/main.cpp  
└── data/ТестовыеДанные

### 5.2. Ключевые классы и функции

## 6. Экспериментальное окружение

**Hardware/OS:**
- **CPU:** Intel Core i5-11400H (6 cores, 12 threads, 2.70 GHz base frequency)
- **RAM:** 16.0 GB DDR4
- **Storage:** SSD 512 GB
- **OS:** Windows 10 

**Toolchain:**
- **Compiler:** Microsoft Visual C++ 2019 (MSVC 19.29.30153)
- **MPI Implementation:** Microsoft MPI Version10.1.12498.52
- **Build System:** CMake 3.30.3
- **Build Type:** Release

**Environment:**
- **PPC_NUM_PROC:** 1, 2, 3, 4, 6 

**Data:**  

## 7. Результаты

### 7.1 Корректность

### 7.2 Производительность

| Mode | Count | Time, s | Speedup | Efficiency |
|------|-------|---------|---------|------------|
| seq  | 1     | 17.20   | 1.00    | N/A        |
| mpi  | 2     | 12.80   | 1.34    | 67.2%      |
| mpi  | 3     | 10.36   | 1.66    | 55.3%      |
| mpi  | 4     | 9.79    | 1.76    | 43.9%      |
| mpi  | 6     | 10.49   | 1.64    | 27.3%      |

**Анализ результатов:**

## 8. Заключение

## 9. Источники
1. Microsoft MPI : документация [Электронный ресурс] // Microsoft Learn. – URL: https://learn.microsoft.com/ru-ru/message-passing-interface/microsoft-mpi (дата обращения: 03.11.2025).
2. Сысоев А. В. Курс лекций по параллельному программированию