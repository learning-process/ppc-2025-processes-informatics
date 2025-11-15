#pragma once

#include "gasenin_l_int_rec_meth/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gasenin_l_int_rec_meth {

struct DiffInfo {
  size_t pos;
  int result;
};

class GaseninLIntRecMethMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GaseninLIntRecMethMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  void RunSequentialComparison(const std::string &str1, const std::string &str2);
  bool RunParallelComparison(const std::string &str1, const std::string &str2, int rank, int size);
};

}  // namespace gasenin_l_int_rec_meth
