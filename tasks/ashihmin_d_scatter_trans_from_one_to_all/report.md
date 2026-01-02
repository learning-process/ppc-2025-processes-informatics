Отчёт

Scatter-распределение от одного ко всем (scatter)

Студент: Ашихмин Даниил
Группа: 3823Б1ФИ1

Технология: SEQ–MPI

------

1. Введение

В данной работе реализуется параллельный алгоритм scatter-распределения - разбиение массива данных на части и рассылка их от одного процесса всем остальным - с использованием технологии MPI.

Цель работы - продемонстрировать корректное распределение данных между процессами с применением tree-based алгоритма, а также провести оценку корректности и производительности решения.

------

2. Постановка задачи

Дан массив data длины N. Требуется разбить его на равные части по elements_per_process элементов и разослать эти части от выбранного процесса root всем остальным процессам, включая сам корневой процесс.

Входные данные
 • Массив data длины
N = elements_per_process × world_size
произвольного типа (int, float, double)
 • Целое число elements_per_process > 0
 • Номер процесса root, с которого выполняется рассылка

Выходные данные
 • У каждого процесса локальный буфер размера elements_per_process, содержащий соответствующую часть исходного массива

Требования
 • Алгоритм должен корректно работать при любом числе процессов
 • Рассылка должна выполняться по дереву с использованием только MPI_Send и MPI_Recv
 • Поддержка трёх типов данных: MPI_INT, MPI_FLOAT, MPI_DOUBLE
 • Последовательная версия должна корректно воспроизводить результат для сравнения

------

3. Базовый алгоритм (последовательный)

Последовательная версия имитирует scatter-операцию:
 • Проверяется, является ли процесс корневым (root == 0)
 • Если процесс корневой, копируются первые elements_per_process элементов в локальный буфер
 • Для некорневых процессов (или при root ≠ 0 в последовательной версии) локальный буфер заполняется нулями

------

4. Схема распараллеливания (MPI)

Распределение данных
 • Процесс root содержит исходный массив размера
elements_per_process × world_size
 • Массив логически разбивается на world_size блоков по elements_per_process элементов
 • Каждый процесс получает свой соответствующий блок

------

Схема взаимодействия процессов

Каждый процесс вычисляет виртуальный ранг:

virtual_rank = (rank - root + size) % size

На каждом шаге используется битовая маска mask.
 • Если (virtual_rank & mask) == 0, процесс отправляет данные потомкам:
 • dest_virtual = virtual_rank | mask
 • dest_real = (dest_virtual + root) % size
Если процесс является корневым, он отправляет соответствующую часть исходного массива, иначе - свой локальный буфер.
 • Иначе процесс получает данные от предка:
 • src_virtual = virtual_rank & (~mask)
 • src_real = (src_virtual + root) % size

После получения данных процесс может участвовать в дальнейшей рассылке.

------

Псевдокод

virtual_rank = (rank - root + size) % size
local_data = new vector[elements_per_process]

mask = 1
while mask < size:
  if (virtual_rank & mask) == 0:
    dest_virtual = virtual_rank | mask
    if dest_virtual < size:
      dest_real = (dest_virtual + root) % size
      if rank == root:
       offset = dest_virtual * elements_per_process
       send data[offset : offset + elements_per_process] to dest_real
      else:
       send local_data to dest_real
  else:
    src_virtual = virtual_rank & (~mask)
    src_real = (src_virtual + root) % size
    recv local_data from src_real
    break
  mask <<= 1

if rank == root:
  offset = virtual_rank * elements_per_process
  copy data[offset : offset + elements_per_process] to local_data

------

Топология
 • Используется стандартный коммуникатор MPI_COMM_WORLD
 • Бинарное дерево, построенное на основе виртуальных рангов

------

5. Детали реализации

Структура кода
 • common.hpp - структура ScatterParams и базовые типы
 • ops_seq.hpp / ops_seq.cpp - последовательная реализация
 • ops_mpi.hpp / ops_mpi.cpp - MPI-реализация (tree-based scatter)
 • tests/functional/main.cpp - функциональные тесты
 • tests/performance/main.cpp - тесты производительности

------

Особенности реализации
 • Вспомогательные функции вынесены в анонимное пространство имён
 • Использованы шаблоны для поддержки разных типов данных