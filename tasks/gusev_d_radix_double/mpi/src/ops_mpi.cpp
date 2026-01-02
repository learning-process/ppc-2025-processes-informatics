#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <cmath>

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

  MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_DOUBLE,
               local_data.data(), send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GusevDRadixDoubleSEQ::RadixSort(local_data);

  std::vector<double> gathered_data;
  if (rank == 0) {
    gathered_data.resize(n);
  }

  MPI_Gatherv(local_data.data(), send_counts[rank], MPI_DOUBLE,
              gathered_data.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().clear();
    GetOutput().reserve(n);
    
    std::vector<double> current_merged;
    
    if (size > 0) {
        auto start = gathered_data.begin();
        auto end = gathered_data.begin() + send_counts[0];
        current_merged.assign(start, end);
    }

    for (int i = 1; i < size; ++i) {
      if (send_counts[i] == 0) continue;
      
      auto start = gathered_data.begin() + displs[i];
      auto end = gathered_data.begin() + displs[i] + send_counts[i];
      
      std::vector<double> temp_res;
      temp_res.reserve(current_merged.size() + (end - start));
      std::merge(current_merged.begin(), current_merged.end(),
                 start, end,
                 std::back_inserter(temp_res));
      
      current_merged = std::move(temp_res);
    }
    GetOutput() = current_merged;
  }

  return true;
}

bool GusevDRadixDoubleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_radix_double