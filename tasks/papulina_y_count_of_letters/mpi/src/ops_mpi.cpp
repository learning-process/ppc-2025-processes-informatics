#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "papulina_y_count_of_letters/common/include/common.hpp"
//#include "util/include/util.hpp"

namespace papulina_y_count_of_letters {

PapulinaYCountOfLettersMPI::PapulinaYCountOfLettersMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procNum_);
}
int PapulinaYCountOfLettersMPI::CountOfLetters(const char *s, const int &n) {
  int k = 0;
  if (n <= 0) {
    return 0;
  }
  for (int i = 0; i < n; i++) {
    char c = s[i];
    if (isalpha(c) != 0) {
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
  int proc_rank = 0;
  int result = 0;
  std::string part_of_string;  // части строки, которая будет обрабатываться потоком
  unsigned int len = 0;        // предполагаемая длина обрабатываемой части
  unsigned int true_len = 0;   // реальная длина обрабатываемой части
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    std::string s = GetInput();

    len = s.size() / procNum_;
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (int i = 1; i < procNum_; i++) {
      unsigned int begin = i * len;
      unsigned int end =
          (i == procNum_ - 1) ? s.size() : begin + len;  // таким образом, если потоков, сильно больше, чем
                                                         // длина строки, то последнему уйдет вся работа
      true_len = end - begin;
      MPI_Send(&true_len, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      if (end - begin != 0) {
        MPI_Send(s.substr(begin, end).data(), static_cast<int>(end - begin), MPI_CHAR, i, 1, MPI_COMM_WORLD);
      } else {
        MPI_Send("", 0, MPI_CHAR, i, 1, MPI_COMM_WORLD);
      }
    }
    true_len = std::min(len, static_cast<unsigned int>(s.size()));
    part_of_string = s.substr(0, true_len);
  } else {
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Recv(&true_len, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    part_of_string.resize(true_len);
    MPI_Recv(part_of_string.data(), static_cast<int>(true_len), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int local_result = CountOfLetters(part_of_string.data(), static_cast<int>(true_len));
  MPI_Reduce(&local_result, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PapulinaYCountOfLettersMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace papulina_y_count_of_letters
