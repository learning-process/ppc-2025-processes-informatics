#include "konstantinov_s_elem_vec_sign_change_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>
// #include <numeric>
#include <cstring>
#include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

KonstantinovSElemVecSignChangeMPI::KonstantinovSElemVecSignChangeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KonstantinovSElemVecSignChangeMPI::ValidationImpl() {
  // std::cout << "\t\tValidation mpi\n";
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
  int step = 0;  // chunk size = step+1
  EType *sendbuf = nullptr;
  int rem = 0;
  int elemcount = 0;  // не пересылается, известен только корню
  if (rank == 0) {
    auto input = GetInput();  // получаем только на нулевом процессе - корне
    if (input.empty()) {
      // std::cout << "GOT EMPTY INPUT!!!!!!!!!!!!!\n";
      return false;
    }
    elemcount = static_cast<int>(input.size());
    // sendbuf = input.data(); //input инвалидируется позже????
    sendbuf = new EType[elemcount];
    std::memcpy(sendbuf, input.data(), input.size() * sizeof(EType));
    //  нужно для перекрывающихся областей pcount= 3 [5] 6/3=2 -> 012 234 4
    step = (elemcount + 1) / pcount;
    rem = elemcount - (step * (pcount - 1));

    // std::cout<<"ROOT got "<<elemcount<<" step, rem = "<<step<<" "<<rem<<"\n";
    // for(int i=0;i<elemcount;i++)
    //   std::cout<<sendbuf[i]<<" ";
    // std::cout<<"\n\n";
  }
  //
  MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);  // корень отправляет, остальные получают
  int chunksz = step + 1;
  int *sendcounts = nullptr;
  int *displs = nullptr;
  EType *recbuf = nullptr;

  if (rank == 0) {
    sendcounts = new int[pcount];
    displs = new int[pcount];
    sendcounts[0] = 0;  // на корень не шлём
    displs[0] = 0;
    // обозначаем перекрывающиеся области (последний элемент = первый в следующем куске)
    for (int i = 1; i < pcount; i++) {
      sendcounts[i] = chunksz;
      displs[i] = (i - 1) * step;
    }
  } else {
    recbuf = new EType[chunksz];  // только некорни выыделяют буфер
  }
  // std::cout<<"Rank "<<rank<<" sendbuf pre and post scatterv\n";
  // for(int i=0;i<elemcount;i++)
  //     std::cout<<sendbuf[i]<<" ";
  //   std::cout<<"\n\n";

  // существуют только буферы нужные получателям/отправителю, ненужные = nullptr (например sendbuf у некорней)
  //rank0: sendbuf, sendcounts, displs
  //rank1+: recbuf
  MPI_Scatterv(sendbuf, sendcounts, displs, MPI_SHORT, recbuf, rank == 0 ? 0 : chunksz, MPI_SHORT, 0, MPI_COMM_WORLD);

  // for(int i=0;i<elemcount;i++)
  //     std::cout<<sendbuf[i]<<" ";
  //   std::cout<<"\n\n";

  int local_res = 0;

  if (rank == 0) {
    delete[] sendcounts;
    delete[] displs;
    if (rem > 1) {
      for (int i = elemcount - rem; i < elemcount - 1; i++) {
        // std::cout<<sendbuf[i];
        local_res += static_cast<int>((sendbuf[i] > 0) != (sendbuf[i + 1] > 0));
      }
      // std::cout<<" root counted = "<<local_res<<"\n";
      // for(int i=0;i<elemcount;i++)
      // std::cout<<sendbuf[i]<<" ";
      // std::cout<<"\n\n";
    }
    
  } else {
    for (int i = 0; i < step; i++) {
      local_res += static_cast<int>((recbuf[i] > 0) != (recbuf[i + 1] > 0));
    }
    // std::cout<<rank<<"# counted = "<<local_res<<"\n";
  }

  //rank0: sendbuf
  //rank1+: recbuf

  if (rank == 0) {
    delete[] sendbuf;
  } else {
    delete[] recbuf;
  }

  //all memory deleted

  int global_res = 0;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // std::cout<<"MPI result: "<<global_res<<" from rank "<<rank<<"\n";
  GetOutput() = global_res;
  return true;
}

bool KonstantinovSElemVecSignChangeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace konstantinov_s_elem_vec_sign_change_count
