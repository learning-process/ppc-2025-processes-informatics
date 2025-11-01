#include "../include/trapezoid_integration_mpi.hpp"
#include <mpi.h>

#include <cmath>
#include <iostream>

namespace kutergin_v_trapezoid_mpi
{

double func(double x) // интегрируемая функция для примера
{
    return x * x;
}

TrapezoidIntegrationMPI::TrapezoidIntegrationMPI(const kutergin_v_trapezoid_seq::InType& in)
{
    SetTypeOfTask(GetStaticTypeOfTask()); // установка типа задачи
    GetInput() = in; // сохранение входных данных
    GetOutput() = 0.0; // инициализация выходных данных
}

bool TrapezoidIntegrationMPI::ValidationImpl()
{
    int process_count; 
    MPI_Comm_size(MPI_COMM_WORLD, &process_count); // получение общего числа процессов

    return (GetInput().b > GetInput().a) && (GetInput().n > 0) && (GetInput().n % process_count == 0); // проверка b > a (границ интегрирования), n > 0 (число разбиений) и число разбиений делится нацело на число процессов
}

bool TrapezoidIntegrationMPI::PreProcessingImpl()
{
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank); // получение ранга процесса 

    kutergin_v_trapezoid_seq::InType tmp_input = GetInput(); // создание хранилища для данных со всех процессов

    MPI_Bcast(&tmp_input, sizeof(tmp_input), MPI_BYTE, 0, MPI_COMM_WORLD); // получение остальными процессами исходных данных от процесса с рангом 0

    GetInput() = tmp_input;

    return true;
}

bool TrapezoidIntegrationMPI::RunImpl()
{
    int process_rank;
    int process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);


    double a = GetInput().a;
    double b = GetInput().b;
    int n = GetInput().n;
    double h = (b - a) / n;
    
    int local_n = n / process_count; // количество трапеций на один процесс
    double local_a = a + process_rank * local_n * h; // начало отрезка для текущего процесса

    // локальные вычисления
    double local_sum = 0.0;
    if (local_n > 0)
    {
        local_sum = (func(local_a) + func(local_a + local_n * h)) / 2.0;
    }

    for (int i = 1; i < n; ++i)
    {
        local_sum += func(local_a + i * h);
    }

    // агрегация 
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (process_rank == 0)
    {
        GetOutput() = global_sum * h;
    }

    return true;
}

bool TrapezoidIntegrationMPI::PostProcessingImpl()
{
    return true;
}


}