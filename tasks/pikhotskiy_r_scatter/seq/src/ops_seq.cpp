#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
// #include <vector>  // Удалено, не используется напрямую

#include "pikhotskiy_r_scatter/common/include/common.hpp"

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterSEQ::PikhotskiyRScatterSEQ(const InputType &input_args) {
  SetTypeOfTask(GetTaskType());
  GetInput() = input_args;
}

bool PikhotskiyRScatterSEQ::ValidationImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &args = GetInput();

  if (args.elements_to_send <= 0) {
    return false;
  }

  if (args.elements_to_send != args.elements_to_receive) {
    return false;
  }

  if (args.send_data_type != args.receive_data_type) {
    return false;
  }

  if (args.source_buffer == nullptr) {
    return false;
  }

  if (args.destination_buffer == nullptr) {
    return false;
  }

  auto check_type = [](MPI_Datatype dt) { return dt == MPI_INT || dt == MPI_FLOAT || dt == MPI_DOUBLE; };

  return check_type(args.send_data_type);
}

bool PikhotskiyRScatterSEQ::PreProcessingImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  return true;
}

bool PikhotskiyRScatterSEQ::RunImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &args = GetInput();

  // Определяем размер одного элемента
  std::size_t element_size = 0;
  if (args.send_data_type == MPI_INT) {
    element_size = sizeof(int);
  } else if (args.send_data_type == MPI_FLOAT) {
    element_size = sizeof(float);
  } else if (args.send_data_type == MPI_DOUBLE) {
    element_size = sizeof(double);
  }

  // Вычисляем общий размер данных в байтах
  std::size_t total_data_size = static_cast<std::size_t>(args.elements_to_send) * element_size;

  // Копируем исходные данные
  auto source_start = static_cast<const std::uint8_t *>(args.source_buffer);
  auto source_end = source_start + total_data_size;

  std::vector<std::uint8_t> processed_data(source_start, source_end);

  // Копируем в буфер назначения
  std::memcpy(args.destination_buffer, processed_data.data(), total_data_size);

  // Сохраняем результат
  GetOutput() = std::move(processed_data);

  return true;
}

bool PikhotskiyRScatterSEQ::PostProcessingImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  return true;
}

}  // namespace pikhotskiy_r_scatter
