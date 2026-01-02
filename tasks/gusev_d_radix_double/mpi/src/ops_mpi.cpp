#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

namespace gusev_d_radix_double {

GusevDRadixDoubleMPI::GusevDRadixDoubleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GusevDRadixDoubleMPI::ValidationImpl() {
  return true;
}

bool GusevDRadixDoubleMPI::PreProcessingImpl() {
  return true;
}

bool GusevDRadixDoubleMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = GetInput().size();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  int remainder = n % size;
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = n / size + (i < remainder ? 1 : 0);
    displs[i] = sum;
    sum += send_counts[i];
  }

  std::vector<double> local_data(send_counts[rank]);

  MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_data.data(), send_counts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GusevDRadixDoubleSEQ::RadixSort(local_data);

  std::vector<double> gathered_data;
  if (rank == 0) {
    gathered_data.resize(n);
  }

  MPI_Gatherv(local_data.data(), send_counts[rank], MPI_DOUBLE, gathered_data.data(), send_counts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    struct Chunk {
      size_t start;
      size_t len;
    };

    std::vector<Chunk> chunks;
    chunks.reserve(size);
    for (int i = 0; i < size; ++i) {
      if (send_counts[i] > 0) {
        chunks.push_back({(size_t)displs[i], (size_t)send_counts[i]});
      }
    }

    while (chunks.size() > 1) {
      std::vector<Chunk> next_chunks;
      next_chunks.reserve((chunks.size() + 1) / 2);

      for (size_t i = 0; i < chunks.size(); i += 2) {
        if (i + 1 < chunks.size()) {
          auto begin = gathered_data.begin() + chunks[i].start;
          auto middle = gathered_data.begin() + chunks[i + 1].start;
          auto end = gathered_data.begin() + chunks[i + 1].start + chunks[i + 1].len;

          std::inplace_merge(begin, middle, end);

          next_chunks.push_back({chunks[i].start, chunks[i].len + chunks[i + 1].len});
        } else {
          next_chunks.push_back(chunks[i]);
        }
      }
      chunks = std::move(next_chunks);
    }

    GetOutput() = std::move(gathered_data);
  }

  return true;
}

bool GusevDRadixDoubleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_radix_double
