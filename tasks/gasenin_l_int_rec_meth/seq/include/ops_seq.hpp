#pragma once

#include "gasenin_l_int_rec_meth/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gasenin_l_int_rec_meth {

class GaseninLIntRecMethSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GaseninLIntRecMethSEQ(const InType &in);

  // Статические методы для работы с пользовательским вводом
  static InType ReadInteractive();
  static InType ReadFromFile(const std::string &filename);
  static void PrintResult(const InType &input, OutType result);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gasenin_l_int_rec_meth
