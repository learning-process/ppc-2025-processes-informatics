#include "romanov_a_crs_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "romanov_a_crs_product/common/include/common.hpp"

namespace romanov_a_crs_product {

RomanovACRSProductMPI::RomanovACRSProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRS(static_cast<uint64_t>(0));
}

bool RomanovACRSProductMPI::ValidationImpl() {
  return (std::get<0>(GetInput()).GetCols() == std::get<1>(GetInput()).GetRows());
}

bool RomanovACRSProductMPI::PreProcessingImpl() {
  return true;
}

inline void BroadcastCRS(CRS &M, int root, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  uint64_t dims[2] = {static_cast<uint64_t>(M.n), static_cast<uint64_t>(M.m)};
  MPI_Bcast(dims, 2, MPI_UINT64_T, root, comm);

  if (rank != root) {
    M.n = static_cast<uint64_t>(dims[0]);
    M.m = static_cast<uint64_t>(dims[1]);
    M.row_index.resize(M.n + 1);
  }

  uint64_t nnz = static_cast<uint64_t>(M.Nnz());
  MPI_Bcast(&nnz, 1, MPI_UINT64_T, root, comm);

  if (rank != root) {
    M.value.resize(static_cast<uint64_t>(nnz));
    M.column.resize(static_cast<uint64_t>(nnz));
  }

  if (nnz > 0) {
    MPI_Bcast(M.value.data(), static_cast<int>(nnz), MPI_DOUBLE, root, comm);
    if (nnz > 0) {
      MPI_Bcast(reinterpret_cast<uint64_t *>(M.column.data()), static_cast<int>(nnz), MPI_UINT64_T, root, comm);
    }
  }

  if (M.n + 1 > 0) {
    MPI_Bcast(reinterpret_cast<uint64_t *>(M.row_index.data()), static_cast<int>(M.n + 1), MPI_UINT64_T, root, comm);
  }
}

inline void SendCRS(const CRS &M, int dest, int tag, MPI_Comm comm) {
  uint64_t dims[2] = {static_cast<uint64_t>(M.n), static_cast<uint64_t>(M.m)};
  MPI_Send(dims, 2, MPI_UINT64_T, dest, tag, comm);

  uint64_t nnz = static_cast<uint64_t>(M.Nnz());
  MPI_Send(&nnz, 1, MPI_UINT64_T, dest, tag + 1, comm);

  if (nnz > 0) {
    MPI_Send(M.value.data(), static_cast<int>(nnz), MPI_DOUBLE, dest, tag + 2, comm);
    if (nnz > 0) {
      MPI_Send(reinterpret_cast<const uint64_t *>(M.column.data()), static_cast<int>(nnz), MPI_UINT64_T, dest, tag + 3,
               comm);
    }
  }

  if (M.n + 1 > 0) {
    MPI_Send(reinterpret_cast<const uint64_t *>(M.row_index.data()), static_cast<int>(M.n + 1), MPI_UINT64_T, dest,
             tag + 4, comm);
  }
}

inline void RecvCRS(CRS &M, int src, int tag, MPI_Comm comm) {
  uint64_t dims[2];
  MPI_Recv(dims, 2, MPI_UINT64_T, src, tag, comm, MPI_STATUS_IGNORE);
  M.n = static_cast<uint64_t>(dims[0]);
  M.m = static_cast<uint64_t>(dims[1]);

  uint64_t nnz;
  MPI_Recv(&nnz, 1, MPI_UINT64_T, src, tag + 1, comm, MPI_STATUS_IGNORE);

  M.value.resize(static_cast<uint64_t>(nnz));
  M.column.resize(static_cast<uint64_t>(nnz));
  M.row_index.resize(M.n + 1);

  if (nnz > 0) {
    MPI_Recv(M.value.data(), static_cast<int>(nnz), MPI_DOUBLE, src, tag + 2, comm, MPI_STATUS_IGNORE);
    if (nnz > 0) {
      MPI_Recv(reinterpret_cast<uint64_t *>(M.column.data()), static_cast<int>(nnz), MPI_UINT64_T, src, tag + 3, comm,
               MPI_STATUS_IGNORE);
    }
  }

  if (M.n + 1 > 0) {
    MPI_Recv(reinterpret_cast<uint64_t *>(M.row_index.data()), static_cast<int>(M.n + 1), MPI_UINT64_T, src, tag + 4,
             comm, MPI_STATUS_IGNORE);
  }
}

bool RomanovACRSProductMPI::RunImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  CRS a, b;

  if (rank == 0) {
    a = std::get<0>(GetInput());
    b = std::get<1>(GetInput());
  }

  BroadcastCRS(b, 0, MPI_COMM_WORLD);

  CRS a_local;

  if (rank == 0) {
    uint64_t n = a.n;
    uint64_t rows_per_proc = (n + num_processes - 1) / num_processes;

    for (int p = 1; p < num_processes; p++) {
      uint64_t start = p * rows_per_proc;
      if (start >= n) {
        CRS empty(0, a.m);
        SendCRS(empty, p, 100 + p, MPI_COMM_WORLD);
        continue;
      }
      uint64_t end = std::min(n, start + rows_per_proc);
      CRS part = a.ExtractRows(start, end);
      SendCRS(part, p, 100 + p, MPI_COMM_WORLD);
    }

    uint64_t end0 = std::min(a.n, rows_per_proc);
    a_local = a.ExtractRows(0, end0);

  } else {
    RecvCRS(a_local, 0, 100 + rank, MPI_COMM_WORLD);
  }

  CRS c_local;
  if (a_local.n > 0 && b.n > 0) {
    c_local = a_local * b;
  } else {
    c_local = CRS(0, b.m);
  }

  CRS c_total;

  if (rank == 0) {
    std::vector<CRS> parts;
    parts.reserve(static_cast<uint64_t>(num_processes));

    parts.push_back(std::move(c_local));

    for (int p = 1; p < num_processes; p++) {
      CRS temp;
      RecvCRS(temp, p, 200 + p, MPI_COMM_WORLD);
      parts.push_back(std::move(temp));
    }

    c_total = CRS::ConcatRows(parts);

  } else {
    SendCRS(c_local, 0, 200 + rank, MPI_COMM_WORLD);
  }

  BroadcastCRS(c_total, 0, MPI_COMM_WORLD);

  GetOutput() = c_total;

  return true;
}

bool RomanovACRSProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
