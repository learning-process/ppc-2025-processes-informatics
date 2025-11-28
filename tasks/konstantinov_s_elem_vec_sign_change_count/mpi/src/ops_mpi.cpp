#include "konstantinov_s_elem_vec_sign_change_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

KonstantinovSElemVecSignChangeMPI::KonstantinovSElemVecSignChangeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KonstantinovSElemVecSignChangeMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KonstantinovSElemVecSignChangeMPI::PreProcessingImpl() {
  return true;
}

bool KonstantinovSElemVecSignChangeMPI::RunImpl() {
  int pcount = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &pcount);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int step = 0; // chunk size = step+1
  int* sendbuf = nullptr;
  int rem = 0;
  int elemcount = 0; //не пересылается, известен только корню
  if(rank == 0)
  {
    auto input = GetInput(); // получаем только на нулевом процессе - корне
    if (input.empty()) {
      return false;
    }
    elemcount = input.size();
    sendbuf = input.data();
    //  нужно для перекрывающихся областей pcount= 3 [5] 6/3=2 -> 012 234 4
    step = (elemcount+1)/pcount; 
    rem = elemcount - step*(pcount-1);
  }

  MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD); //корень отправляет, остальные получают
  int chunksz = step+1;
  int* sendcounts = nullptr;
  int* displs = nullptr;
  int* recbuf = nullptr;

  if(rank == 0)
  {
    sendcounts = new int[pcount];
    displs = new int[pcount];
    sendcounts[0] = 0; //на корень не шлём
    displs[0] = 0;
    // обозначаем перекрывающиеся области (последний элемент = первый в следующем куске)
    for(int i=1;i<pcount;i++)
    {
      sendcounts[i] = chunksz;
      displs[i] = (i-1)*step;
    }
  }
  else
  {
    recbuf = new int[chunksz]; //только некорни выыделяют буфер
  }
  // существуют только буферы нужные получателям/отправителю, ненужные = nullptr (например sendbuf у некорней)
  MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recbuf,
                rank == 0 ? 0 : chunksz, MPI_INT, 0, MPI_COMM_WORLD);
  
  int local_res = 0;

  if(rank != 0)
  {
    for(int i=0; i<step; i++)
      local_res+= (recbuf[i]>0)!=(recbuf[i+1]>0);
  }
  else
  {
    if (rem>1)
    {
      for(int i=elemcount-rem; i<elemcount-1; i++)
        local_res+= (sendbuf[i]>0)!=(sendbuf[i+1]>0);
    }
  }

  int global_res = 0;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(rank == 0)
    {
      delete[] sendcounts;
      delete[] displs;
    }
    else
    {
      delete[] recbuf;
    }

  GetOutput() = global_res;
  return true;
}

bool KonstantinovSElemVecSignChangeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace konstantinov_s_elem_vec_sign_change_count
