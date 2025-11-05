#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "util/include/util.hpp"

namespace papulina_y_count_of_letters {

PapulinaYCountOfLettersMPI::PapulinaYCountOfLettersMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procNum_);
}
int PapulinaYCountOfLettersMPI::CountOfLetters(const char *s, const int &n) {
  int k = 0;
  for (int i = 0; i < n; i++) {
    char c = s[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      k++;
    }
  }
  return k;
}
bool PapulinaYCountOfLettersMPI::ValidationImpl() {
  return procNum_ > 0;
}

bool PapulinaYCountOfLettersMPI::PreProcessingImpl() {
  return true;
}

bool PapulinaYCountOfLettersMPI::RunImpl() {
  std::cout << "Input data in RunImpl: " << GetInput() << std::endl;
  int procRank = 0;
  int result = 0;
  std::string partOfString = "";  // части строки, которая будет обрабатываться потоком
  unsigned int len = 0;           // предполагаемая длина обрабатываемой части
  unsigned int trueLen = 0;       // реальная длина обрабатываемой части
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  if (procRank == 0) {
    std::string s = GetInput();

    len = s.size() / procNum_;
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (int i = 1; i < procNum_; i++) {
      int begin = i * len;
      int end = (i == procNum_ - 1) ? s.size() : begin + len;  // таким образом, если потоков, сильно больше, чем
                                                               // длина строки, то последнему уйдет вся работа
      trueLen = end - begin;
      MPI_Send(&trueLen, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      if (end - begin != 0) {
        MPI_Send(s.substr(begin, end).data(), end - begin, MPI_CHAR, i, 1, MPI_COMM_WORLD);
      } else {
        MPI_Send("", 0, MPI_CHAR, i, 1, MPI_COMM_WORLD);
      }
    }
    trueLen = std::min(len, (unsigned int)s.size());
    partOfString = s.substr(0, trueLen);
  } else {
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Recv(&trueLen, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    partOfString.resize(trueLen);
    MPI_Recv(&partOfString[0], trueLen, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int localResult = CountOfLetters(partOfString.data(), trueLen);
  MPI_Reduce(&localResult, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (procRank == 0) {
    GetOutput() = result;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Output data in RunImpl: " << GetOutput() << std::endl;
  return true;
}

bool PapulinaYCountOfLettersMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace papulina_y_count_of_letters
