#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EgorovaLGaussFilterVertMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ApplyFilter(const std::vector<uint8_t> &local_in, std::vector<uint8_t> &local_out, int rows,
                          int local_cols, int total_lc, int ch);

  static void FillLocalWithHalo(const std::vector<uint8_t> &local_data, std::vector<uint8_t> &local_with_halo, int rows,
                                int local_cols, int total_lc, int left_h, int ch);

  static void ExchangeHalo(std::vector<uint8_t> &local_with_halo, int target, int rows, int total_lc, int ch,
                           size_t send_col, size_t recv_col);
};

}  // namespace egorova_l_gauss_filter_vert
