  #include "baldin_a_gauss_filter/mpi/include/ops_mpi.hpp"

  #include <mpi.h>

  #include <numeric>
  #include <vector>

  #include "baldin_a_gauss_filter/common/include/common.hpp"
  #include "util/include/util.hpp"

  namespace baldin_a_gauss_filter {

  BaldinAGaussFilterMPI::BaldinAGaussFilterMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
  }

  bool BaldinAGaussFilterMPI::ValidationImpl() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        const auto& im = GetInput();
        return (im.width > 0 && im.height > 0 && im.channels > 0 && 
                im.pixels.size() == static_cast<size_t>(im.width * im.height * im.channels));
    }
    return true;
  }

  bool BaldinAGaussFilterMPI::PreProcessingImpl() {
      return true;
  }

  bool BaldinAGaussFilterMPI::RunImpl() {
    ImageData &input = GetInput();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 0, height = 0, channels = 0;

    if (rank == 0) {
        width = input.width;
        height = input.height;
        channels = input.channels;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int rows_per_proc = height / size;
    const int remainder = height % size;

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    std::vector<int> real_counts(size);

    int current_global_row = 0;
    int row_size_bytes = width * channels;

    for (int i = 0; i < size; ++i) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        real_counts[i] = rows;
        
        int send_start_row = current_global_row - 1; 
        int send_end_row   = current_global_row + rows; // +1 строка снизу (индекс)

        if (send_start_row < 0) send_start_row = 0;
        if (send_end_row >= height) send_end_row = height - 1;

        int rows_to_send = send_end_row - send_start_row + 1;
        
        counts[i] = rows_to_send * row_size_bytes;
        displs[i] = send_start_row * row_size_bytes;

        current_global_row += rows;
    }

    int my_real_rows = real_counts[rank];
    int my_recv_rows = counts[rank] / row_size_bytes;

    // Входящий буфер (uint8_t)
    std::vector<uint8_t> local_buffer(my_recv_rows * row_size_bytes);
    
    // Промежуточный буфер должен быть больше (uint16_t или int),
    // так как мы храним сумму без деления (до 1020).
    std::vector<uint16_t> horiz_buffer(my_recv_rows * width * channels); 

    std::vector<uint8_t> result_buffer(my_real_rows * row_size_bytes);

    MPI_Scatterv(rank == 0 ? input.pixels.data() : nullptr, 
                counts.data(), displs.data(), MPI_UINT8_T,
                local_buffer.data(), counts[rank], MPI_UINT8_T,
                0, MPI_COMM_WORLD);

    const int kernel[3] = {1, 2, 1}; 
    
    // Горизонтальный проход
    for (int y = 0; y < my_recv_rows; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = -1; k <= 1; ++k) {
                    int nx = std::clamp(x + k, 0, width - 1);
                    sum += local_buffer[(y * width + nx) * channels + c] * kernel[k + 1];
                }
                horiz_buffer[(y * width + x) * channels + c] = static_cast<uint16_t>(sum);
            }
        }
    }

    int row_offset = (rank == 0) ? 0 : 1;

    // Вертикальный проход
    for (int i = 0; i < my_real_rows; ++i) {
        int local_y = row_offset + i;

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = -1; k <= 1; ++k) {
                    
                    int neighbor_y = local_y + k;
                    if (neighbor_y < 0) neighbor_y = 0; 
                    if (neighbor_y >= my_recv_rows) neighbor_y = my_recv_rows - 1;

                    sum += horiz_buffer[(neighbor_y * width + x) * channels + c] * kernel[k + 1];
                }

                result_buffer[(i * width + x) * channels + c] = static_cast<uint8_t>(sum / 16);
            }
        }
    }

    std::vector<int> recv_counts(size);
    std::vector<int> recv_displs(size);
    

    GetOutput().width = width;
    GetOutput().height = height;
    GetOutput().channels = channels;
    GetOutput().pixels.resize(width * height * channels);

    int current_disp = 0;
    for(int i=0; i<size; ++i) {
        recv_counts[i] = real_counts[i] * row_size_bytes;
        recv_displs[i] = current_disp;
        current_disp += recv_counts[i];
    }

    MPI_Allgatherv(result_buffer.data(), my_real_rows * row_size_bytes, MPI_UINT8_T,
                GetOutput().pixels.data(), 
                recv_counts.data(), recv_displs.data(), MPI_UINT8_T,
                MPI_COMM_WORLD);

    return true;
  }

  bool BaldinAGaussFilterMPI::PostProcessingImpl() {
    
    return true;
  }

  }  // namespace baldin_a_gauss_filter
