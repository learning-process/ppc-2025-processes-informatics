#pragma once

#include "task/include/task.hpp"
#include <vector>
#include <string>
#include <tuple>


namespace kutergin_v_reduce
{
    
struct InputData 
{
    std::vector<int> data;
    int root = 0; // root по умолчанию
};

using InType = InputData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<int, int, std::string>;

}