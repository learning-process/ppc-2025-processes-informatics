#include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_my_scatter {

BaldinAMyScatterMPI::BaldinAMyScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAMyScatterMPI::ValidationImpl() {
  const auto& input = GetInput();

  const auto& [sendbuf, sendcount, sendtype, 
                 recvbuf, recvcount, recvtype, 
                 root, comm] = input;

  int world_size = 0;
  MPI_Comm_size(comm, &world_size);

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return (sendcount > 0 && 
            sendcount == recvcount && 
            sendtype == recvtype && // Должны совпадать
            is_sup_type(sendtype) && 
            root >= 0);
}

  bool BaldinAMyScatterMPI::PreProcessingImpl() {
    // auto& input = GetInput();
    // const auto& [sendbuf_old, sendcount, sendtype, recvbuf_old, recvcount_dummy, recvtype, root, comm] = input;
    // int world_size = 0;
    // MPI_Comm_size(comm, &world_size);
    // std::get<6>(input) = (root % world_size);
    //int world_size = 0;
    //MPI_Comm_size(comm, world_size);

    //int total_cnt = world_size * sendcount;
    //MPI_Datatype type_to_use = sendtype;

    //std::vector<int> tmp = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    //std::get<0>(input) = static_cast<const void*>(tmp.data());
    auto& input = GetInput();
    
    const auto& [sendbuf_d, sendcount_d, sendtype_d, recvbuf_d, recvcount_d, recvtype_d, root, comm] = input;
    
    int world_size = 0;
    MPI_Comm_size(comm, &world_size);
    
    // если root выходит за границы, корректируем его
    if (root >= world_size) {
        std::get<6>(input) = root % world_size; 
    }

    return true;
  }

bool BaldinAMyScatterMPI::RunImpl() {
  auto& input = GetInput();
  const auto& [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Aint lb, send_extent, recv_extent;
    
  // Определяем шаг данных в байтах
  if (rank == root) {
      MPI_Type_get_extent(sendtype, &lb, &send_extent);
  } else {
      MPI_Type_get_extent(recvtype, &lb, &recv_extent);
      send_extent = recv_extent; 
  }

  // Виртуальный ранг: root становится 0
  int v_rank = (rank - root + size) % size;

  const char* curr_buf_ptr = nullptr;
  std::vector<char> temp_buffer;

  // --- ЭТАП 1: Подготовка (только на root) ---
  if (rank == root) {
      size_t total_bytes = (size_t)size * sendcount * send_extent;
      size_t chunk_bytes = (size_t)sendcount * send_extent;
      
      try {
          temp_buffer.resize(total_bytes);
      } catch(const std::bad_alloc&) {
          MPI_Abort(comm, MPI_ERR_NO_MEM);
      }

      const char* send_ptr = static_cast<const char*>(sendbuf);
      char* tmp_ptr = temp_buffer.data();

      // Циклический сдвиг: копируем данные так, чтобы блок для Root был первым
      // [0][1][2][3], root=2 => [2][3][0][1]
      
      size_t first_part_bytes = (size - root) * chunk_bytes;
      size_t second_part_bytes = root * chunk_bytes;

      // Часть от root до конца -> в начало temp_buffer
      std::memcpy(tmp_ptr, send_ptr + second_part_bytes, first_part_bytes);
      // Часть от 0 до root -> в конец temp_buffer
      std::memcpy(tmp_ptr + first_part_bytes, send_ptr, second_part_bytes);

      curr_buf_ptr = temp_buffer.data();
  }

  // Ближайшая степень двойки для маски
  int mask = 1;
  while (mask < size) mask <<= 1;
  mask >>= 1;

  // --- ЭТАП 2: Рассылка по дереву ---
  while (mask > 0) {
      // Если процесс - отправитель на этом уровне
      if (v_rank % (2 * mask) == 0) {
          int v_dest = v_rank + mask;

          if (v_dest < size) {
              int real_dest = (v_dest + root) % size; // Обратное преобразование в физ. ранг

              // Размер поддерева получателя
              int subtree_size = v_dest + mask; 
              if (subtree_size > size) subtree_size = size;
              
              int count_to_send = (subtree_size - v_dest) * recvcount;
              size_t offset_bytes = (size_t)(v_dest - v_rank) * recvcount * send_extent;

              MPI_Send(curr_buf_ptr + offset_bytes, count_to_send, 
                        (rank == root ? sendtype : recvtype), 
                        real_dest, 0, comm);
          }
      }
      // Если процесс - получатель
      else if (v_rank % (2 * mask) == mask) {
          int v_source = v_rank - mask;
          int real_source = (v_source + root) % size;

          int subtree_end = v_rank + mask;
          if (subtree_end > size) subtree_end = size;
          
          int count_to_recv = (subtree_end - v_rank) * recvcount;
          size_t bytes_to_recv = count_to_recv * send_extent;

          // Выделяем память под входящие данные
          temp_buffer.resize(bytes_to_recv); 

          MPI_Recv(temp_buffer.data(), count_to_recv, recvtype, real_source, 0, comm, MPI_STATUS_IGNORE);
          
          // Теперь работаем с полученным буфером
          curr_buf_ptr = temp_buffer.data();
      }

      mask >>= 1;
  }

  // --- ЭТАП 3: Копирование в пользовательский буфер ---
  if (recvbuf != MPI_IN_PLACE) {
      // Копируем только свою долю (recvcount)
      std::memcpy(recvbuf, curr_buf_ptr, recvcount * send_extent);
  }
  GetOutput() = recvbuf;
  return true;
}

bool BaldinAMyScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_my_scatter