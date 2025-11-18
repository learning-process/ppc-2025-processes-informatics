#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {
    VolkovACountWordLineSEQ::VolkovACountWordLineSEQ(const InType& in) : BaseTask(in) {}
    bool VolkovACountWordLineSEQ::ValidationImpl() { 
        return true; 
    }
    bool VolkovACountWordLineSEQ::PreProcessingImpl() { 
        return true; 
    }
    bool VolkovACountWordLineSEQ::RunImpl() { 
        /* compute */ 
        return true; }
    bool VolkovACountWordLineSEQ::PostProcessingImpl() { 
        /* write result */ 
        return true; }
}