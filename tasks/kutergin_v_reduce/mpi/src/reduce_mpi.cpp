#include "../include/reduce_mpi.hpp"

#include <mpi.h>

#include <cstring>

#include "../../common/include/common.hpp"

namespace kutergin_v_reduce {

ReduceMPI::ReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0;                       // инициализация выходных данных
}

int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  int process_rank = 0;
  int process_count = 0;
  MPI_Comm_rank(comm, &process_rank);
  MPI_Comm_size(comm, &process_count);

  int type_size = 0;
  MPI_Type_size(datatype, &type_size);

  // Древовидный сбор
  for (int mask = 1; mask < process_count;
       mask <<= 1)  // удвоение битовой маски на каждой итерации посредством битового сдвига
  {
    if ((process_rank & mask) != 0)  // процессы-отправители
    {
      MPI_Send(sendbuf, count, datatype, process_rank - mask, 0, comm);
      break;
    } else if (process_rank + mask < process_count)  // процессы-получатели
    {
      auto *recv_temp = new uint8_t[count * type_size];
      MPI_Recv(recv_temp, count, datatype, process_rank + mask, 0, comm, MPI_STATUS_IGNORE);

      if (op == MPI_SUM && datatype == MPI_INT)  // выполнение MPI_SUM для int
      {
        for (int i = 0; i < count; ++i) {
          reinterpret_cast<int *>(sendbuf)[i] += reinterpret_cast<int *>(recv_temp)[i];
        }
      }

      delete[] recv_temp;
    }
  }

  // Результат с процесса 0 отправляется на процесс 'root'
  if (process_rank == 0 && root != 0) {
    MPI_Send(sendbuf, count, datatype, root, 0, comm);
  }

  // Корневой процесс 'root' получает финальный результат
  if (process_rank == root) {
    if (root == 0) {
      std::memcpy(recvbuf, sendbuf, count * type_size);
    } else {
      MPI_Recv(recvbuf, count, datatype, 0, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  return MPI_SUCCESS;
}

bool ReduceMPI::ValidationImpl() {
  return true;
}

bool ReduceMPI::PreProcessingImpl() {
  return true;
}

bool ReduceMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &input = GetInput();
  int root_process = input.root;
  const auto &input_vec = input.data;

  int send_data = input_vec.empty() ? 0 : input_vec[0];  // у каждого процесса - свое число
  int recv_data = 0;                                     // буфер для результат

  // Вызов своей реализации Reduce()
  Reduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, root_process, MPI_COMM_WORLD);

  // Только корневой процесс записывает результат в Output
  if (rank == root_process) {
    GetOutput() = recv_data;
  }

  return true;
}

bool ReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_reduce
