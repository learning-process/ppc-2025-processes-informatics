#pragma once

#include <vector>
#include <mpi.h>

#include "khruev_a_global_opt/common/include/common.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptMPI : public BaseTask {
public:
    explicit TestTaskMPI(const InType& in);
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;

private:
    std::vector<Trial> trials_;
    OutType result_;

    double CalculateFunction(double t);
    void AddTrialUnsorted(double t, double z);
};

} // namespace strongin_method