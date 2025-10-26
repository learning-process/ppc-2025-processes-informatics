#pragma once

#include "task/include/task.hpp"

namespace kutergin_v_trapezoid_seq
{
  
struct InputData
{
    double a;
    double b;
    int n;
};

using InType = InputData;
using OutType = double; 
using BaseTask = ppc::task::Task<InType, OutType>;

}